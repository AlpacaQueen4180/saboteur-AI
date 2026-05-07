package main;

import model.Board;
import model.Cell;
import model.GameLogicController;
import model.GoalType;
import model.Player;
import model.Position;
import model.Tool;
import model.cards.Card;
import model.cards.PathCard;
import model.cards.PlayerActionCard;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class RlFeatureExtractor {
  private RlFeatureExtractor() {}

  public static Map<String, Object> pathFeatures(
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    Map<String, Object> features = new LinkedHashMap<>();

    double[] distances = distancesToGoals(board);
    double averageDistance = (distances[0] + distances[1] + distances[2]) / 3.0;

    int knownGoldIndex = knownGoldIndex(knownGoals);
    boolean goldKnown = knownGoldIndex >= 0;
    double distanceToKnownGold = goldKnown ? distances[knownGoldIndex] : -1.0;
    double targetDistance = goldKnown ? distanceToKnownGold : averageDistance;

    List<String> knownGoalsList = knownGoalsList(knownGoals);

    features.put("gold_known", goldKnown);
    features.put("goldKnown", goldKnown);

    features.put("known_gold_index", knownGoldIndex);
    features.put("knownGoldIndex", knownGoldIndex);

    features.put("known_goals", knownGoalsList);
    features.put("knownGoals", knownGoalsList);

    features.put("distance_to_top_goal", distances[0]);
    features.put("distanceToTopGoal", distances[0]);

    features.put("distance_to_middle_goal", distances[1]);
    features.put("distanceToMiddleGoal", distances[1]);

    features.put("distance_to_bottom_goal", distances[2]);
    features.put("distanceToBottomGoal", distances[2]);

    features.put("average_distance_to_all_goals", averageDistance);
    features.put("averageDistanceToAllGoals", averageDistance);

    features.put("distance_to_known_gold", distanceToKnownGold);
    features.put("distanceToKnownGold", distanceToKnownGold);

    features.put("target_distance", targetDistance);
    features.put("targetDistance", targetDistance);

    features.put("frontier_count", frontierPositions(board).size());
    features.put("frontierCount", frontierPositions(board).size());

    features.put("reachable_count", board.getReachable().size());
    features.put("reachableCount", board.getReachable().size());

    features.put("destroyable_count", board.getDestroyable().size());
    features.put("destroyableCount", board.getDestroyable().size());

    return features;
  }

  public static void enrichBoardDto(
      Map<String, Object> boardDto,
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    Map<String, Object> pathFeatures = pathFeatures(board, knownGoals);

    boardDto.put("reachable_count", board.getReachable().size());
    boardDto.put("reachableCount", board.getReachable().size());

    boardDto.put("destroyable_count", board.getDestroyable().size());
    boardDto.put("destroyableCount", board.getDestroyable().size());

    boardDto.put("path_features", pathFeatures);
    boardDto.put("pathFeatures", pathFeatures);
  }

  public static List<Map<String, Object>> enrichLegalActions(
      List<Map<String, Object>> rawActions,
      GameLogicController game,
      Player controlledPlayer,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    List<Map<String, Object>> enriched = new ArrayList<>();
    List<Card> hand = controlledPlayer.visibleHand();

    for (int actionId = 0; actionId < rawActions.size(); actionId++) {
      Map<String, Object> raw = rawActions.get(actionId);
      Map<String, Object> action = new LinkedHashMap<>(raw);

      String type = String.valueOf(action.get("type"));
      int handIndex = intValue(action.get("handIndex"), -1);

      Card card = null;
      if (handIndex >= 0 && handIndex < hand.size()) {
        card = hand.get(handIndex);
      }

      action.put("action_id", actionId);
      action.put("actionId", actionId);

      action.put("move_type", type);
      action.put("moveType", type);

      if (card != null) {
        action.put("card_name", card.name());
        action.put("cardName", card.name());

        action.put("card_type", card.type().name());
        action.put("cardType", card.type().name());

        action.put("card_id", card.id());
        action.put("cardId", card.id());

        if (card instanceof PathCard) {
          PathCard pathCard = (PathCard) card;
          action.put("path_type", pathCard.pathType().name());
          action.put("pathType", pathCard.pathType().name());
          action.put("card_sides", sidesDto(pathCard.sides()));
          action.put("cardSides", sidesDto(pathCard.sides()));
        }

        if (card instanceof PlayerActionCard) {
          PlayerActionCard playerActionCard = (PlayerActionCard) card;
          action.put("player_action_type", playerActionCard.playerActionType().name());
          action.put("playerActionType", playerActionCard.playerActionType().name());
          action.put("effects", toolsDto(playerActionCard.effects()));
        }
      } else {
        action.put("card_name", null);
        action.put("cardName", null);
        action.put("card_type", null);
        action.put("cardType", null);
        action.put("card_id", -1);
        action.put("cardId", -1);
      }

      attachDefaultActionFields(action, game.board(), knownGoals);
      attachActionArgs(action, type);

      if ("PLAY_PATH".equals(type) && card instanceof PathCard) {
        attachPathSimulationFields(action, game.board(), knownGoals, (PathCard) card);
      } else if ("PLAY_ROCKFALL".equals(type)) {
        attachRockfallSimulationFields(action, game.board(), knownGoals);
      } else if ("PLAY_PLAYER".equals(type)) {
        int targetPlayer = intValue(action.get("targetPlayer"), -1);
        action.put("target_player", targetPlayer);
      } else if ("PLAY_MAP".equals(type)) {
        int goalIndex = goalIndex(String.valueOf(action.get("goal")));
        action.put("goal_index", goalIndex);
        action.put("goalIndex", goalIndex);
      }

      enriched.add(action);
    }

    return enriched;
  }

  private static void attachDefaultActionFields(
      Map<String, Object> action,
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    double beforeTarget = targetDistance(board, knownGoals);

    action.putIfAbsent("x", -1);
    action.putIfAbsent("y", -1);
    action.putIfAbsent("rotated", false);

    action.put("target_player", -1);
    action.putIfAbsent("targetPlayer", -1);

    action.put("goal_index", -1);
    action.putIfAbsent("goalIndex", -1);

    action.put("before_target_distance", beforeTarget);
    action.put("beforeTargetDistance", beforeTarget);

    action.put("after_target_distance", beforeTarget);
    action.put("afterTargetDistance", beforeTarget);

    action.put("delta_target_distance", 0.0);
    action.put("deltaTargetDistance", 0.0);

    action.put("after_reachable_count", board.getReachable().size());
    action.put("afterReachableCount", board.getReachable().size());

    action.put("after_destroyable_count", board.getDestroyable().size());
    action.put("afterDestroyableCount", board.getDestroyable().size());

    action.put("after_remove_distance", beforeTarget);
    action.put("afterRemoveDistance", beforeTarget);

    action.put("after_ideal_fill_distance", beforeTarget);
    action.put("afterIdealFillDistance", beforeTarget);

    action.put("remove_delta", 0.0);
    action.put("removeDelta", 0.0);

    action.put("ideal_fill_delta", 0.0);
    action.put("idealFillDelta", 0.0);
  }

  private static void attachActionArgs(Map<String, Object> action, String type) {
    List<Object> args = new ArrayList<>();

    if ("PLAY_PATH".equals(type)) {
      int x = intValue(action.get("x"), -1);
      int y = intValue(action.get("y"), -1);
      boolean rotated = boolValue(action.get("rotated"), false);
      args.add(x);
      args.add(y);
      args.add(rotated ? 1 : 0);
    } else if ("PLAY_ROCKFALL".equals(type)) {
      args.add(intValue(action.get("x"), -1));
      args.add(intValue(action.get("y"), -1));
    } else if ("PLAY_PLAYER".equals(type)) {
      args.add(intValue(action.get("targetPlayer"), -1));
    } else if ("PLAY_MAP".equals(type)) {
      args.add(goalIndex(String.valueOf(action.get("goal"))));
    }

    action.put("args", args);
  }

  private static void attachPathSimulationFields(
      Map<String, Object> action,
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals,
      PathCard originalCard
  ) {
    int x = intValue(action.get("x"), -1);
    int y = intValue(action.get("y"), -1);
    boolean rotated = boolValue(action.get("rotated"), false);

    double beforeTarget = targetDistance(board, knownGoals);
    double afterTarget = beforeTarget;
    int afterReachable = board.getReachable().size();
    int afterDestroyable = board.getDestroyable().size();

    PathCard card = (PathCard) originalCard.copy();
    card.setRotated(rotated);

    Board simulated = board.simulatePlaceCardAt(card, x, y);
    if (simulated != null) {
      afterTarget = targetDistance(simulated, knownGoals);
      afterReachable = simulated.getReachable().size();
      afterDestroyable = simulated.getDestroyable().size();
    }

    double delta = beforeTarget - afterTarget;

    action.put("x", x);
    action.put("y", y);
    action.put("rotated", rotated);

    action.put("before_target_distance", beforeTarget);
    action.put("beforeTargetDistance", beforeTarget);

    action.put("after_target_distance", afterTarget);
    action.put("afterTargetDistance", afterTarget);

    action.put("delta_target_distance", delta);
    action.put("deltaTargetDistance", delta);

    action.put("after_reachable_count", afterReachable);
    action.put("afterReachableCount", afterReachable);

    action.put("after_destroyable_count", afterDestroyable);
    action.put("afterDestroyableCount", afterDestroyable);
  }

  private static void attachRockfallSimulationFields(
      Map<String, Object> action,
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    int x = intValue(action.get("x"), -1);
    int y = intValue(action.get("y"), -1);

    double beforeTarget = targetDistance(board, knownGoals);
    double afterRemoveDistance = beforeTarget;
    double afterIdealFillDistance = beforeTarget;

    int afterReachable = board.getReachable().size();
    int afterDestroyable = board.getDestroyable().size();

    Board afterRemove = board.simulateRemoveCardAt(x, y);
    if (afterRemove != null) {
      afterRemoveDistance = targetDistance(afterRemove, knownGoals);
      afterReachable = afterRemove.getReachable().size();
      afterDestroyable = afterRemove.getDestroyable().size();

      PathCard ideal = new PathCard(-9999, PathCard.Type.CROSSROAD_PATH);
      ideal.setRotated(false);

      Board afterIdealFill = afterRemove.simulatePlaceCardAt(ideal, x, y);
      if (afterIdealFill != null) {
        afterIdealFillDistance = targetDistance(afterIdealFill, knownGoals);
      } else {
        afterIdealFillDistance = afterRemoveDistance;
      }
    }

    double deltaTargetDistance = beforeTarget - afterRemoveDistance;
    double removeDelta = afterRemoveDistance - beforeTarget;
    double idealFillDelta = beforeTarget - afterIdealFillDistance;

    action.put("x", x);
    action.put("y", y);

    action.put("before_target_distance", beforeTarget);
    action.put("beforeTargetDistance", beforeTarget);

    action.put("after_target_distance", afterRemoveDistance);
    action.put("afterTargetDistance", afterRemoveDistance);

    action.put("delta_target_distance", deltaTargetDistance);
    action.put("deltaTargetDistance", deltaTargetDistance);

    action.put("after_reachable_count", afterReachable);
    action.put("afterReachableCount", afterReachable);

    action.put("after_destroyable_count", afterDestroyable);
    action.put("afterDestroyableCount", afterDestroyable);

    action.put("after_remove_distance", afterRemoveDistance);
    action.put("afterRemoveDistance", afterRemoveDistance);

    action.put("after_ideal_fill_distance", afterIdealFillDistance);
    action.put("afterIdealFillDistance", afterIdealFillDistance);

    action.put("remove_delta", removeDelta);
    action.put("removeDelta", removeDelta);

    action.put("ideal_fill_delta", idealFillDelta);
    action.put("idealFillDelta", idealFillDelta);
  }

  private static double targetDistance(
      Board board,
      Map<Board.GoalPosition, GoalType> knownGoals
  ) {
    double[] distances = distancesToGoals(board);
    int goldIndex = knownGoldIndex(knownGoals);

    if (goldIndex >= 0) {
      return distances[goldIndex];
    }

    return (distances[0] + distances[1] + distances[2]) / 3.0;
  }

  private static double[] distancesToGoals(Board board) {
    Position[] goals = new Position[] {
        board.topGoalPosition(),
        board.middleGoalPosition(),
        board.bottomGoalPosition()
    };

    List<Position> sources = frontierPositions(board);
    double[] distances = new double[3];

    for (int i = 0; i < 3; i++) {
      distances[i] = minManhattanDistance(sources, goals[i]);
    }

    return distances;
  }

  private static List<Position> frontierPositions(Board board) {
    Set<Position> reachable = board.getReachable();
    List<Position> frontier = new ArrayList<>();

    for (Position p : reachable) {
      Cell cell = board.cellAt(p);
      if (cell != null && !cell.hasCard()) {
        frontier.add(p);
      }
    }

    if (frontier.isEmpty()) {
      frontier.addAll(reachable);
    }

    return frontier;
  }

  private static double minManhattanDistance(List<Position> sources, Position target) {
    if (sources == null || sources.isEmpty() || target == null) {
      return -1.0;
    }

    int best = Integer.MAX_VALUE;

    for (Position p : sources) {
      int d = Math.abs(p.x - target.x) + Math.abs(p.y - target.y);
      if (d < best) {
        best = d;
      }
    }

    return best == Integer.MAX_VALUE ? -1.0 : best;
  }

  private static int knownGoldIndex(Map<Board.GoalPosition, GoalType> knownGoals) {
    if (knownGoals == null) return -1;

    if (knownGoals.get(Board.GoalPosition.TOP) == GoalType.GOLD) return 0;
    if (knownGoals.get(Board.GoalPosition.MIDDLE) == GoalType.GOLD) return 1;
    if (knownGoals.get(Board.GoalPosition.BOTTOM) == GoalType.GOLD) return 2;

    return -1;
  }

  private static List<String> knownGoalsList(Map<Board.GoalPosition, GoalType> knownGoals) {
    List<String> result = new ArrayList<>();

    for (Board.GoalPosition position : Board.GoalPosition.values()) {
      GoalType type = knownGoals == null ? null : knownGoals.get(position);
      result.add(type == null ? "UNKNOWN" : type.name());
    }

    return result;
  }

  private static int goalIndex(String goal) {
    if (goal == null) return -1;

    String normalized = goal.toUpperCase();
    if ("TOP".equals(normalized)) return 0;
    if ("MIDDLE".equals(normalized)) return 1;
    if ("BOTTOM".equals(normalized)) return 2;

    return -1;
  }

  private static int intValue(Object value, int defaultValue) {
    if (value == null) return defaultValue;

    if (value instanceof Number) {
      return ((Number) value).intValue();
    }

    try {
      return Integer.parseInt(String.valueOf(value));
    } catch (Exception e) {
      return defaultValue;
    }
  }

  private static boolean boolValue(Object value, boolean defaultValue) {
    if (value == null) return defaultValue;

    if (value instanceof Boolean) {
      return (Boolean) value;
    }

    return Boolean.parseBoolean(String.valueOf(value));
  }

  private static List<String> sidesDto(PathCard.Side[] sides) {
    List<String> dto = new ArrayList<>();
    Arrays.stream(sides).forEach(side -> dto.add(side.name()));
    return dto;
  }

  private static List<String> toolsDto(Tool[] tools) {
    List<String> dto = new ArrayList<>();
    for (Tool tool : tools) {
      dto.add(tool.name());
    }
    return dto;
  }
}