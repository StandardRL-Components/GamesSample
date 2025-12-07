import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based physics game where players shoot a particle to score points. "
        "Use colored pads for boosts and upgrade your abilities between rounds."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to aim and ←→ to adjust power. Press space to shoot. "
        "In the upgrade menu, use ←→ to select and space to confirm."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_ROUNDS = 5
        self.MAX_STEPS_PER_EPISODE = 2000  # Safety break

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_PLAYER1 = (0, 150, 255)
        self.COLOR_PLAYER2 = (255, 100, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_ACCENT = (255, 200, 0)
        self.COLOR_RED_ZONE = (255, 50, 50)
        self.COLOR_GREEN_ZONE = (50, 255, 50)
        self.COLOR_BLUE_ZONE = (50, 50, 255)
        self.COLOR_WHITE = (255, 255, 255)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.game_over = False
        self.phase = None
        self.current_player = 0
        self.round = 0
        self.scores = None
        self.player_upgrades = None
        self.particle = None
        self.aim_angle = None
        self.aim_power = None
        self.last_space_held = False
        self.upgrade_selection = 0
        self.round_score_awarded = False
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False
        self.phase = 'AIMING'
        self.current_player = 0
        self.round = 1
        self.scores = [0, 0]
        self.player_upgrades = [
            {"power": 1.0, "friction": 0.98, "boost": 1.5},  # Player 1
            {"power": 1.0, "friction": 0.98, "boost": 1.5}   # Player 2
        ]
        self.particle = None
        self.aim_angle = -math.pi / 2
        self.aim_power = 5.0
        self.last_space_held = False
        self.upgrade_selection = 0
        self.round_score_awarded = False
        self.last_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if self.phase == 'AIMING':
            self._handle_aiming_input(movement, space_pressed)

        elif self.phase == 'SHOOTING':
            reward += self._update_particle()

        elif self.phase == 'UPGRADE':
            self._handle_upgrade_input(movement, space_pressed)

        elif self.phase == 'GAME_OVER':
            if space_pressed:
                # This doesn't auto-reset, but allows agent to see final screen.
                # The environment runner (e.g. stable-baselines3) will call reset().
                pass

        if self.steps >= self.MAX_STEPS_PER_EPISODE:
            terminated = True
        
        # The game ends when the phase is GAME_OVER, after the final scores are tallied.
        if self.game_over:
            terminated = True

        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_aiming_input(self, movement, space_pressed):
        # Adjust Angle (Up/Down)
        if movement == 1: self.aim_angle -= 0.05
        if movement == 2: self.aim_angle += 0.05
        self.aim_angle = max(-math.pi, min(0, self.aim_angle))

        # Adjust Power (Left/Right)
        if movement == 3: self.aim_power -= 0.2
        if movement == 4: self.aim_power += 0.2
        self.aim_power = max(1.0, min(15.0, self.aim_power))

        if space_pressed:
            # SFX: Launch particle
            upgrades = self.player_upgrades[self.current_player]
            power = self.aim_power * upgrades["power"]
            self.particle = {
                "pos": np.array([self.WIDTH / 2, self.HEIGHT - 50.0]),
                "vel": np.array([math.cos(self.aim_angle) * power, math.sin(self.aim_angle) * power]),
                "radius": 8,
                "color": self.COLOR_PLAYER1 if self.current_player == 0 else self.COLOR_PLAYER2,
                "player": self.current_player,
                "friction": upgrades["friction"],
                "boost": upgrades["boost"],
                "trail": []
            }
            self.phase = 'SHOOTING'

    def _update_particle(self):
        reward = 0
        if self.particle is None: return reward

        # Update trail
        self.particle["trail"].append(self.particle["pos"].copy())
        if len(self.particle["trail"]) > 50:
            self.particle["trail"].pop(0)

        # Apply velocity
        self.particle["pos"] += self.particle["vel"]

        # Apply friction
        self.particle["vel"] *= self.particle["friction"]

        # Boundary checks and bounces
        pos, vel, r = self.particle["pos"], self.particle["vel"], self.particle["radius"]
        if pos[0] - r < 0 or pos[0] + r > self.WIDTH:
            vel[0] *= -0.8  # SFX: Wall bounce
            pos[0] = np.clip(pos[0], r, self.WIDTH - r)
        if pos[1] - r < 0 or pos[1] + r > self.HEIGHT:
            vel[1] *= -0.8  # SFX: Wall bounce
            pos[1] = np.clip(pos[1], r, self.HEIGHT - r)

        # Check for color boosts
        reward += self._check_color_zones()

        # Check if stopped
        if np.linalg.norm(self.particle["vel"]) < 0.05:
            # SFX: Particle stop
            reward += self._evaluate_turn()
            self.particle = None  # Clear the particle after evaluation

        return reward

    def _check_color_zones(self):
        reward = 0
        player_color = self.particle["color"]
        pos = self.particle["pos"]

        zones = [
            (self.COLOR_RED_ZONE, pygame.Rect(50, 150, 100, 100)),
            (self.COLOR_GREEN_ZONE, pygame.Rect(self.WIDTH / 2 - 50, 50, 100, 100)),
            (self.COLOR_BLUE_ZONE, pygame.Rect(self.WIDTH - 150, 150, 100, 100))
        ]

        for color, rect in zones:
            if rect.collidepoint(pos[0], pos[1]):
                match = False
                if color == self.COLOR_RED_ZONE and player_color == self.COLOR_PLAYER2: match = True
                if color == self.COLOR_BLUE_ZONE and player_color == self.COLOR_PLAYER1: match = True
                # No green player, so green zone is neutral or for future use

                if match:
                    # SFX: Boost activate
                    self.particle["vel"] *= self.particle["boost"]
                    reward += 1.0  # Event-based reward for boost
                    # Cap speed to prevent instability
                    speed = np.linalg.norm(self.particle["vel"])
                    if speed > 25.0:
                        self.particle["vel"] = (self.particle["vel"] / speed) * 25.0
                break  # Only one boost at a time
        return reward

    def _evaluate_turn(self):
        reward = 0
        if self.particle is None: return reward

        # Calculate score based on final position
        dist_from_center = np.linalg.norm(self.particle["pos"] - np.array([self.WIDTH / 2, 75]))

        points = 0
        if dist_from_center < 25: points = 100  # Bullseye
        elif dist_from_center < 50: points = 50
        elif dist_from_center < 100: points = 20
        elif dist_from_center < 150: points = 10

        self.scores[self.current_player] += points

        # Continuous reward for proximity
        reward += max(0, 1 - (dist_from_center / (self.WIDTH / 2))) * 0.5

        # Switch player or end round
        if self.current_player == 0:
            self.current_player = 1
            self.phase = 'AIMING'
            self._reset_aim()
        else:
            self.current_player = 0
            
            # End of round reward
            if self.scores[0] > self.scores[1]: reward += 10
            elif self.scores[1] > self.scores[0]: reward -= 10

            if self.round >= self.MAX_ROUNDS:
                self.phase = 'GAME_OVER'
                self.game_over = True
            else:
                self.phase = 'UPGRADE'

        return reward

    def _handle_upgrade_input(self, movement, space_pressed):
        # Select Upgrade (Left/Right)
        if movement == 3: self.upgrade_selection = (self.upgrade_selection - 1 + 3) % 3
        if movement == 4: self.upgrade_selection = (self.upgrade_selection + 1) % 3

        if space_pressed:
            # SFX: Upgrade confirm
            # Apply upgrades for both players
            for i in range(2):
                upgrades = self.player_upgrades[i]
                if self.upgrade_selection == 0: upgrades["power"] += 0.05
                elif self.upgrade_selection == 1: upgrades["friction"] = max(0.9, upgrades["friction"] - 0.005)
                elif self.upgrade_selection == 2: upgrades["boost"] += 0.1

            # Start next round
            self.round += 1
            self.phase = 'AIMING'
            self._reset_aim()

    def _reset_aim(self):
        self.aim_angle = -math.pi / 2
        self.aim_power = 5.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score_p1": self.scores[0],
            "score_p2": self.scores[1],
            "steps": self.steps,
            "round": self.round,
            "phase": self.phase,
            "current_player": self.current_player
        }

    def _render_game(self):
        self._draw_background()
        self._draw_rink()

        if self.particle:
            self._draw_particle(self.particle)

        if self.phase == 'AIMING':
            self._draw_aim_assists()

    def _draw_background(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _draw_rink(self):
        center_x, center_y = self.WIDTH // 2, 75
        # Scoring zones
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 150, (40, 50, 70))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 150, (60, 70, 90))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 100, (50, 60, 80))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 100, (70, 80, 100))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 50, (60, 70, 90))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 50, (80, 90, 110))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 25, (70, 80, 100))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 25, (90, 100, 120))

        # Color boost zones
        pygame.draw.rect(self.screen, self.COLOR_RED_ZONE, (50, 150, 100, 100), border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_GREEN_ZONE, (self.WIDTH / 2 - 50, 50, 100, 100), border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_BLUE_ZONE, (self.WIDTH - 150, 150, 100, 100), border_radius=10)

    def _draw_particle(self, p):
        # Draw trail
        if len(p["trail"]) > 1:
            for i in range(len(p["trail"]) - 1):
                alpha = int(255 * (i / len(p["trail"])))
                color = (*p["color"], alpha)
                start_pos = (int(p["trail"][i][0]), int(p["trail"][i][1]))
                end_pos = (int(p["trail"][i + 1][0]), int(p["trail"][i + 1][1]))
                # Create a temporary surface for drawing with alpha
                line_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surf, color, start_pos, end_pos, max(1, int(p["radius"] * (i/len(p["trail"])) * 2)))
                self.screen.blit(line_surf, (0,0))


        # Draw glow
        pos = (int(p["pos"][0]), int(p["pos"][1]))
        for i in range(int(p["radius"]), 0, -2):
            alpha = int(100 * (1 - i / p["radius"]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*p["color"], alpha))

        # Draw core particle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"] / 2), p["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p["radius"] / 2), self.COLOR_WHITE)

    def _draw_aim_assists(self):
        start_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50.0])

        # Draw prediction line
        upgrades = self.player_upgrades[self.current_player]
        sim_pos = start_pos.copy()
        sim_vel = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * self.aim_power * upgrades["power"]

        points = [tuple(sim_pos.astype(int))]
        for _ in range(100):
            sim_pos += sim_vel
            sim_vel *= upgrades["friction"]
            if np.linalg.norm(sim_vel) < 0.1: break
            if not (0 < sim_pos[0] < self.WIDTH and 0 < sim_pos[1] < self.HEIGHT): break
            points.append(tuple(sim_pos.astype(int)))

        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_UI_ACCENT, False, points)

        # Draw power bar
        power_width = (self.aim_power / 15.0) * 150
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH / 2 - 75, self.HEIGHT - 25, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (self.WIDTH / 2 - 75, self.HEIGHT - 25, power_width, 15))

    def _render_ui(self):
        # Scores
        p1_color = self.COLOR_PLAYER1 if self.current_player == 0 and self.phase in ['AIMING', 'SHOOTING'] else self.COLOR_UI_TEXT
        p2_color = self.COLOR_PLAYER2 if self.current_player == 1 and self.phase in ['AIMING', 'SHOOTING'] else self.COLOR_UI_TEXT

        p1_score_surf = self.font_medium.render(f"P1: {self.scores[0]}", True, p1_color)
        p2_score_surf = self.font_medium.render(f"P2: {self.scores[1]}", True, p2_color)
        self.screen.blit(p1_score_surf, (20, 10))
        self.screen.blit(p2_score_surf, (self.WIDTH - p2_score_surf.get_width() - 20, 10))

        # Round
        round_surf = self.font_medium.render(f"Round {self.round}/{self.MAX_ROUNDS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(round_surf, (self.WIDTH / 2 - round_surf.get_width() / 2, 10))

        # Phase specific UI
        if self.phase == 'UPGRADE':
            self._draw_upgrade_menu()
        elif self.phase == 'GAME_OVER':
            self._draw_game_over_screen()

    def _draw_upgrade_menu(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))

        title_surf = self.font_large.render("UPGRADE PHASE", True, self.COLOR_UI_ACCENT)
        overlay.blit(title_surf, (self.WIDTH / 2 - title_surf.get_width() / 2, 50))

        options = ["Power", "Friction", "Boost"]
        for i, option in enumerate(options):
            color = self.COLOR_UI_ACCENT if i == self.upgrade_selection else self.COLOR_UI_TEXT
            text_surf = self.font_medium.render(option, True, color)
            overlay.blit(text_surf, (self.WIDTH / 2 - text_surf.get_width() / 2, 150 + i * 50))

        self.screen.blit(overlay, (0, 0))

    def _draw_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))

        winner_text = "TIE"
        if self.scores[0] > self.scores[1]: winner_text = "PLAYER 1 WINS!"
        elif self.scores[1] > self.scores[0]: winner_text = "PLAYER 2 WINS!"

        title_surf = self.font_large.render("GAME OVER", True, self.COLOR_UI_ACCENT)
        winner_surf = self.font_medium.render(winner_text, True, self.COLOR_WHITE)
        reset_surf = self.font_small.render("Press SPACE to continue", True, self.COLOR_UI_TEXT)

        overlay.blit(title_surf, (self.WIDTH / 2 - title_surf.get_width() / 2, 100))
        overlay.blit(winner_surf, (self.WIDTH / 2 - winner_surf.get_width() / 2, 180))
        overlay.blit(reset_surf, (self.WIDTH / 2 - reset_surf.get_width() / 2, 250))
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will open a pygame window for rendering
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # quit the dummy driver
    pygame.init() # re-init with default driver

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Quantum Curling")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False

    # Game loop for manual play
    while not terminated and not truncated:
        movement = 0  # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over. Final Scores: P1={info['score_p1']}, P2={info['score_p2']}")
            # To play again, we need to wait for a moment and then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False


        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()