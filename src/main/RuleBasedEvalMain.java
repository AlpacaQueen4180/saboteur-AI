package main;

import model.*;
import model.cards.Card;

public class RuleBasedEvalMain extends GameObserver {
    private static int miners = 0;
    private static int saboteurs = 0;

    @Override
    protected void onGameFinished(Player.Role role, int lastPlayer) {
        if (role == Player.Role.SABOTEUR) {
            saboteurs++;
        } else {
            miners++;
        }
    }

    @Override
    protected void onPlayerMove(Move move, Card newCard) {
        game().finalizeTurn();
    }

    public static void main(String[] args) throws GameException {
        int runs = 1000;

        if (args.length >= 1) {
            runs = Integer.parseInt(args[0]);
        }

        miners = 0;
        saboteurs = 0;

        for (int i = 0; i < runs; i++) {
            GameState state = new GameState();

            HeuristicsAI ai1 = new HeuristicsAI("Heuristic-0");
            HeuristicsAI ai2 = new HeuristicsAI("Heuristic-1");
            HeuristicsAI ai3 = new HeuristicsAI("Heuristic-2");
            HeuristicsAI ai4 = new HeuristicsAI("Heuristic-3");

            GameLogicController game = new GameLogicController(
                state,
                ai1,
                ai2,
                ai3,
                ai4
            );

            game.addObserver(new RuleBasedEvalMain());
            game.initializeRound();
            game.startRound();
        }

        int total = miners + saboteurs;
        double minerWinRate = total > 0 ? (double) miners / total : 0.0;
        double saboteurWinRate = total > 0 ? (double) saboteurs / total : 0.0;

        System.out.println("========== Rule-based Evaluation ==========");
        System.out.println("players: 4 rule-based HeuristicsAI");
        System.out.println("games: " + total);
        System.out.println("miners wins: " + miners);
        System.out.println("saboteurs wins: " + saboteurs);
        System.out.printf("miner win rate: %.4f%n", minerWinRate);
        System.out.printf("saboteur win rate: %.4f%n", saboteurWinRate);
        System.out.println("===========================================");
    }
}