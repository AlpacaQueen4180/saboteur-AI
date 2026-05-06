package main;

import ai.AI;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import model.Board;
import model.Cell;
import model.GameLogicController;
import model.GameObserver;
import model.GameState;
import model.GoalType;
import model.Move;
import model.Player;
import model.Position;
import model.Tool;
import model.cards.BoardActionCard;
import model.cards.Card;
import model.cards.PathCard;
import model.cards.PlayerActionCard;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;

public class ServerMain {
  private static final int DEFAULT_PORT = 8000;
  private static final int CONTROLLED_PLAYER = 3;
  private static final int NUM_PLAYERS = 4;

  public static void main(String[] args) throws IOException {
    int port = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_PORT;
    GameServer gameServer = new GameServer();
    HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
    server.createContext("/health", gameServer::handleHealth);
    server.createContext("/reset", gameServer::handleReset);
    server.createContext("/state", gameServer::handleState);
    server.createContext("/legal-actions", gameServer::handleLegalActions);
    server.createContext("/step", gameServer::handleStep);
    server.setExecutor(Executors.newSingleThreadExecutor());
    server.start();
    System.out.printf("Saboteur server listening on http://localhost:%d%n", port);
  }

  private static final class GameServer {
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private GameLogicController game;
    private RemotePlayer controlledPlayer;
    private ServerObserver observer;

    void handleHealth(HttpExchange exchange) throws IOException {
      if (!requireMethod(exchange, "GET")) return;
      Map<String, Object> body = new LinkedHashMap<>();
      body.put("ok", true);
      body.put("started", game != null && game.started());
      body.put("finished", game != null && game.finished());
      writeJson(exchange, 200, body);
    }

    synchronized void handleReset(HttpExchange exchange) throws IOException {
      if (!requireMethod(exchange, "POST")) return;
      try {
        observer = new ServerObserver();
        controlledPlayer = new RemotePlayer("Python");
        game = new GameLogicController(
          new GameState(),
          new HeuristicsAI("Heuristic-0"),
          new HeuristicsAI("Heuristic-1"),
          new HeuristicsAI("Heuristic-2"),
          controlledPlayer
        );
        game.addObserver(observer);
        observer.beginTransition();
        game.initializeRound();
        game.startRound();
        writeJson(exchange, 200, buildResponse(0));
      } catch (Exception e) {
        writeError(exchange, 400, e.getMessage());
      }
    }

    synchronized void handleState(HttpExchange exchange) throws IOException {
      if (!requireMethod(exchange, "GET")) return;
      if (!requireGame(exchange)) return;

      String view = queryParams(exchange).getOrDefault("view", "both").toLowerCase(Locale.ROOT);
      Map<String, Object> response = new LinkedHashMap<>();
      if (view.equals("observation") || view.equals("both")) {
        response.put("observation", buildObservation());
      }
      if (view.equals("full") || view.equals("both")) {
        response.put("fullState", buildFullState());
      }
      if (!view.equals("observation") && !view.equals("full") && !view.equals("both")) {
        writeError(exchange, 400, "view must be observation, full, or both");
        return;
      }
      writeJson(exchange, 200, response);
    }

    synchronized void handleLegalActions(HttpExchange exchange) throws IOException {
      if (!requireMethod(exchange, "GET")) return;
      if (!requireGame(exchange)) return;
      writeJson(exchange, 200, buildLegalActions());
    }

    synchronized void handleStep(HttpExchange exchange) throws IOException {
      if (!requireMethod(exchange, "POST")) return;
      if (!requireGame(exchange)) return;
      if (game.finished()) {
        writeError(exchange, 409, "Game is finished");
        return;
      }
      if (game.currentPlayerIndex() != CONTROLLED_PLAYER) {
        writeError(exchange, 409, "It is not player 3's turn");
        return;
      }

      try {
        JsonObject body = readJsonObject(exchange);
        Move move = parseMove(body);
        Player.Role controlledRole = controlledPlayer.visibleRole();
        observer.beginTransition();
        game.playMove(move);
        int reward = terminalReward(controlledRole);
        writeJson(exchange, 200, buildResponse(reward));
      } catch (IllegalArgumentException e) {
        writeError(exchange, 400, e.getMessage());
      } catch (Exception e) {
        writeError(exchange, 409, e.getMessage());
      }
    }

    private boolean requireGame(HttpExchange exchange) throws IOException {
      if (game == null) {
        writeError(exchange, 409, "Game has not been reset");
        return false;
      }
      return true;
    }

    private Map<String, Object> buildResponse(int reward) {
      Map<String, Object> response = new LinkedHashMap<>();
      response.put("observation", buildObservation());
      response.put("fullState", buildFullState());
      response.put("legalActions", buildLegalActions());
      response.put("reward", reward);
      response.put("done", game.finished());
      response.put("winner", observer.winnerName());
      return response;
    }

    private int terminalReward(Player.Role controlledRole) {
      if (!game.finished() || observer.winner == null) return 0;
      return observer.winner == controlledRole ? 1 : -1;
    }

    private List<Map<String, Object>> buildLegalActions() {
      if (game == null || game.finished() || game.currentPlayerIndex() != CONTROLLED_PLAYER) {
        return Collections.emptyList();
      }

      List<Map<String, Object>> actions = new ArrayList<>();
      List<Card> hand = controlledPlayer.visibleHand();
      for (int handIndex = 0; handIndex < hand.size(); handIndex++) {
        Card card = hand.get(handIndex);
        actions.add(action("DISCARD", handIndex));

        if (card instanceof PathCard && !controlledPlayer.isSabotaged()) {
          addPathActions(actions, handIndex, (PathCard) card.copy());
        } else if (card instanceof PlayerActionCard) {
          addPlayerActions(actions, handIndex, (PlayerActionCard) card);
        } else if (card.type() == Card.Type.MAP) {
          for (Board.GoalPosition pos : Board.GoalPosition.values()) {
            Map<String, Object> action = action("PLAY_MAP", handIndex);
            action.put("goal", pos.name());
            actions.add(action);
          }
        } else if (card.type() == Card.Type.ROCKFALL) {
          for (Position position : sortedPositions(game.board().getDestroyable())) {
            Map<String, Object> action = action("PLAY_ROCKFALL", handIndex);
            action.put("x", position.x);
            action.put("y", position.y);
            actions.add(action);
          }
        }
      }
      return actions;
    }

    private void addPathActions(List<Map<String, Object>> actions, int handIndex, PathCard card) {
      card.setRotated(false);
      for (Position position : sortedPositions(game.board().getPlaceable(card))) {
        Map<String, Object> action = action("PLAY_PATH", handIndex);
        action.put("x", position.x);
        action.put("y", position.y);
        action.put("rotated", false);
        actions.add(action);
      }

      card.setRotated(true);
      for (Position position : sortedPositions(game.board().getPlaceable(card))) {
        Map<String, Object> action = action("PLAY_PATH", handIndex);
        action.put("x", position.x);
        action.put("y", position.y);
        action.put("rotated", true);
        actions.add(action);
      }
    }

    private void addPlayerActions(List<Map<String, Object>> actions, int handIndex, PlayerActionCard card) {
      for (int target = 0; target < game.numPlayers(); target++) {
        Player player = game.playerAt(target);
        if (card.type() == Card.Type.BLOCK && target != CONTROLLED_PLAYER && player.isSabotageable(card.effects()[0])) {
          Map<String, Object> action = action("PLAY_PLAYER", handIndex);
          action.put("targetPlayer", target);
          actions.add(action);
        } else if (card.type() == Card.Type.REPAIR && player.isRepairable(card.effects())) {
          Map<String, Object> action = action("PLAY_PLAYER", handIndex);
          action.put("targetPlayer", target);
          actions.add(action);
        }
      }
    }

    private Map<String, Object> action(String type, int handIndex) {
      Map<String, Object> action = new LinkedHashMap<>();
      action.put("type", type);
      action.put("playerIndex", CONTROLLED_PLAYER);
      action.put("handIndex", handIndex);
      return action;
    }

    private Move parseMove(JsonObject body) {
      String type = requiredString(body, "type").toUpperCase(Locale.ROOT);
      int handIndex = requiredInt(body, "handIndex");
      switch (type) {
        case "DISCARD":
          return Move.NewDiscardMove(CONTROLLED_PLAYER, handIndex);
        case "PLAY_PATH":
          return Move.NewPathMove(
            CONTROLLED_PLAYER,
            handIndex,
            requiredInt(body, "x"),
            requiredInt(body, "y"),
            optionalBoolean(body, "rotated", false)
          );
        case "PLAY_PLAYER":
          return Move.NewPlayerActionMove(CONTROLLED_PLAYER, handIndex, requiredInt(body, "targetPlayer"));
        case "PLAY_MAP":
          return Move.NewMapMove(CONTROLLED_PLAYER, handIndex, parseGoal(requiredString(body, "goal")));
        case "PLAY_ROCKFALL":
          return Move.NewRockfallMove(CONTROLLED_PLAYER, handIndex, requiredInt(body, "x"), requiredInt(body, "y"));
        default:
          throw new IllegalArgumentException("Unknown move type: " + type);
      }
    }

    private Board.GoalPosition parseGoal(String goal) {
      try {
        return Board.GoalPosition.valueOf(goal.toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException("goal must be TOP, MIDDLE, or BOTTOM");
      }
    }

    private Map<String, Object> buildObservation() {
      Map<String, Object> observation = new LinkedHashMap<>();
      observation.put("board", boardDto(game.board()));
      observation.put("revealedGoals", revealedGoalsDto(controlledPlayer.knownGoalsView()));
      observation.put("connectivity", connectivityDto());
      observation.put("private", privateDto());
      observation.put("public", publicDto());
      observation.put("playerStatus", playersStatusDto(false));
      observation.put("events", observer.latestEventsDto());
      return observation;
    }

    private Map<String, Object> buildFullState() {
      Map<String, Object> fullState = new LinkedHashMap<>();
      fullState.put("started", game.started());
      fullState.put("finished", game.finished());
      fullState.put("winner", observer.winnerName());
      fullState.put("controlledPlayer", CONTROLLED_PLAYER);
      fullState.put("currentPlayer", game.currentPlayerIndex());
      fullState.put("drawPileSize", game.drawPileSize());
      fullState.put("board", boardDto(game.board()));
      fullState.put("players", playersStatusDto(true));
      fullState.put("legalActions", buildLegalActions());
      return fullState;
    }

    private Map<String, Object> privateDto() {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("playerIndex", CONTROLLED_PLAYER);
      dto.put("role", controlledPlayer.visibleRole().name());
      dto.put("hand", cardsDto(controlledPlayer.visibleHand()));
      return dto;
    }

    private Map<String, Object> publicDto() {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("currentPlayer", game.currentPlayerIndex());
      dto.put("deckSize", game.drawPileSize());
      dto.put("numPlayers", game.numPlayers());
      dto.put("finished", game.finished());
      dto.put("winner", observer.winnerName());
      dto.put("playerStatus", playersStatusDto(false));
      return dto;
    }

    private Map<String, Object> connectivityDto() {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("reachable", positionsDto(game.board().getReachable()));
      dto.put("destroyable", positionsDto(game.board().getDestroyable()));

      List<Map<String, Object>> placeableByCard = new ArrayList<>();
      List<Card> hand = controlledPlayer.visibleHand();
      for (int handIndex = 0; handIndex < hand.size(); handIndex++) {
        Card card = hand.get(handIndex);
        if (!(card instanceof PathCard) || controlledPlayer.isSabotaged()) continue;
        PathCard pathCard = (PathCard) card.copy();
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("handIndex", handIndex);
        item.put("card", cardDto(card));
        pathCard.setRotated(false);
        item.put("normal", positionsDto(game.board().getPlaceable(pathCard)));
        pathCard.setRotated(true);
        item.put("rotated", positionsDto(game.board().getPlaceable(pathCard)));
        placeableByCard.add(item);
      }
      dto.put("placeableByHandCard", placeableByCard);
      return dto;
    }

    private List<Map<String, Object>> playersStatusDto(boolean includePrivate) {
      List<Map<String, Object>> players = new ArrayList<>();
      for (int i = 0; i < game.numPlayers(); i++) {
        Player player = game.playerAt(i);
        Map<String, Object> dto = new LinkedHashMap<>();
        dto.put("index", i);
        dto.put("name", player.name());
        dto.put("handSize", player.handSize());
        dto.put("sabotaged", player.isSabotaged());
        dto.put("blockedTools", toolsDto(player.sabotaged()));
        if (includePrivate) {
          dto.put("role", player.visibleRole().name());
          dto.put("hand", cardsDto(player.visibleHand()));
          dto.put("discarded", cardsDto(player.visibleDiscarded()));
        }
        players.add(dto);
      }
      return players;
    }

    private Map<String, Object> boardDto(Board board) {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("width", board.width());
      dto.put("height", board.height());
      dto.put("start", positionDto(board.startPosition()));
      Map<String, Object> goals = new LinkedHashMap<>();
      goals.put("TOP", positionDto(board.topGoalPosition()));
      goals.put("MIDDLE", positionDto(board.middleGoalPosition()));
      goals.put("BOTTOM", positionDto(board.bottomGoalPosition()));
      dto.put("goals", goals);

      List<Map<String, Object>> cells = new ArrayList<>();
      for (int y = 0; y < board.height(); y++) {
        for (int x = 0; x < board.width(); x++) {
          Cell cell = board.cellAt(x, y);
          Map<String, Object> cellDto = new LinkedHashMap<>();
          cellDto.put("x", x);
          cellDto.put("y", y);
          cellDto.put("hasCard", cell.hasCard());
          cellDto.put("sides", sidesDto(cell.sides()));
          cellDto.put("card", cardDto(cell.card()));
          cells.add(cellDto);
        }
      }
      dto.put("cells", cells);
      return dto;
    }

    private Map<String, Object> revealedGoalsDto(Map<Board.GoalPosition, GoalType> knownGoals) {
      Map<String, Object> dto = new LinkedHashMap<>();
      for (Board.GoalPosition position : Board.GoalPosition.values()) {
        GoalType type = knownGoals.get(position);
        if (type != null) dto.put(position.name(), type.name());
      }
      return dto;
    }

    private List<Map<String, Object>> cardsDto(List<Card> cards) {
      List<Map<String, Object>> dto = new ArrayList<>();
      for (Card card : cards) dto.add(cardDto(card));
      return dto;
    }

    private Map<String, Object> cardDto(Card card) {
      if (card == null) return null;
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("id", card.id());
      dto.put("name", card.name());
      dto.put("type", card.type().name());
      if (card instanceof PathCard) {
        PathCard pathCard = (PathCard) card;
        dto.put("pathType", pathCard.pathType().name());
        dto.put("rotated", pathCard.rotated());
        dto.put("sides", Arrays.stream(pathCard.sides()).map(Enum::name).toArray(String[]::new));
      } else if (card instanceof PlayerActionCard) {
        PlayerActionCard playerActionCard = (PlayerActionCard) card;
        dto.put("playerActionType", playerActionCard.playerActionType().name());
        dto.put("effects", toolsDto(playerActionCard.effects()));
      } else if (card instanceof BoardActionCard) {
        dto.put("boardActionType", ((BoardActionCard) card).boardActionType().name());
      }
      return dto;
    }

    private Map<String, Object> moveDto(Move move) {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("type", move.type().name());
      dto.put("playerIndex", move.playerIndex());
      dto.put("handIndex", move.handIndex());
      dto.put("args", move.args());
      dto.put("card", cardDto(move.card()));
      return dto;
    }

    private Map<String, Object> positionDto(Position position) {
      Map<String, Object> dto = new LinkedHashMap<>();
      dto.put("x", position.x);
      dto.put("y", position.y);
      return dto;
    }

    private List<Map<String, Object>> positionsDto(Set<Position> positions) {
      List<Map<String, Object>> dto = new ArrayList<>();
      for (Position position : sortedPositions(positions)) dto.add(positionDto(position));
      return dto;
    }

    private List<Position> sortedPositions(Set<Position> positions) {
      List<Position> sorted = new ArrayList<>(positions);
      sorted.sort((a, b) -> a.y == b.y ? Integer.compare(a.x, b.x) : Integer.compare(a.y, b.y));
      return sorted;
    }

    private List<String> sidesDto(Cell.Side[] sides) {
      List<String> dto = new ArrayList<>();
      for (Cell.Side side : sides) dto.add(side.name());
      return dto;
    }

    private List<String> toolsDto(Tool[] tools) {
      List<String> dto = new ArrayList<>();
      for (Tool tool : tools) dto.add(tool.name());
      Collections.sort(dto);
      return dto;
    }

    private JsonObject readJsonObject(HttpExchange exchange) throws IOException {
      String body;
      try (InputStream input = exchange.getRequestBody()) {
        body = new String(input.readAllBytes(), StandardCharsets.UTF_8).trim();
      }
      if (body.isEmpty()) return new JsonObject();
      try {
        return JsonParser.parseString(body).getAsJsonObject();
      } catch (Exception e) {
        throw new IllegalArgumentException("Request body must be a JSON object");
      }
    }

    private String requiredString(JsonObject body, String field) {
      if (!body.has(field) || body.get(field).isJsonNull()) {
        throw new IllegalArgumentException("Missing field: " + field);
      }
      return body.get(field).getAsString();
    }

    private int requiredInt(JsonObject body, String field) {
      if (!body.has(field) || body.get(field).isJsonNull()) {
        throw new IllegalArgumentException("Missing field: " + field);
      }
      return body.get(field).getAsInt();
    }

    private boolean optionalBoolean(JsonObject body, String field, boolean defaultValue) {
      return body.has(field) && !body.get(field).isJsonNull() ? body.get(field).getAsBoolean() : defaultValue;
    }

    private Map<String, String> queryParams(HttpExchange exchange) {
      Map<String, String> params = new HashMap<>();
      String query = exchange.getRequestURI().getRawQuery();
      if (query == null || query.isEmpty()) return params;
      for (String part : query.split("&")) {
        String[] pair = part.split("=", 2);
        params.put(pair[0], pair.length > 1 ? pair[1] : "");
      }
      return params;
    }

    private boolean requireMethod(HttpExchange exchange, String method) throws IOException {
      if (exchange.getRequestMethod().equalsIgnoreCase(method)) return true;
      writeError(exchange, 405, "Method must be " + method);
      return false;
    }

    private void writeError(HttpExchange exchange, int status, String message) throws IOException {
      Map<String, Object> body = new LinkedHashMap<>();
      body.put("error", message == null ? "Unknown error" : message);
      writeJson(exchange, status, body);
    }

    private void writeJson(HttpExchange exchange, int status, Object body) throws IOException {
      byte[] bytes = gson.toJson(body).getBytes(StandardCharsets.UTF_8);
      exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
      exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
      exchange.sendResponseHeaders(status, bytes.length);
      try (OutputStream output = exchange.getResponseBody()) {
        output.write(bytes);
      }
    }

    private final class ServerObserver extends GameObserver {
      private final List<Move> latestEvents = new ArrayList<>();
      private Player.Role winner;
      private int lastPlayer = -1;

      void beginTransition() {
        latestEvents.clear();
      }

      @Override
      protected void onPlayerMove(Move move, Card newCard) {
        latestEvents.add(move);
        game().finalizeTurn();
      }

      @Override
      protected void onGameFinished(Player.Role role, int lastPlayer) {
        this.winner = role;
        this.lastPlayer = lastPlayer;
      }

      String winnerName() {
        return winner == null ? null : winner.name();
      }

      List<Map<String, Object>> latestEventsDto() {
        List<Map<String, Object>> dto = new ArrayList<>();
        for (Move move : latestEvents) dto.add(moveDto(move));
        return dto;
      }
    }
  }

  private static final class RemotePlayer extends Player {
    RemotePlayer(String name) {
      super(name);
    }

    @Override
    protected void onMovementPrompt() {
      // Python supplies this player's move through POST /step.
    }

    Map<Board.GoalPosition, GoalType> knownGoalsView() {
      return new LinkedHashMap<>(knownGoals());
    }
  }
}
