
# Generated: 2025-08-27T12:28:33.511845
# Source Brief: brief_00056.md
# Brief Index: 56

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A rhythm-based roguelike Gymnasium environment.

    The player navigates a grid, battling enemies to the beat of an internal
    metronome. Actions are taken on the beat, and timing is crucial for
    success. The goal is to clear 5 levels of increasing difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move on the grid. Space to attack adjacent cells. "
        "All actions occur on the beat."
    )

    # User-facing description of the game
    game_description = (
        "A rhythm-based roguelike. Move and attack on the beat to clear "
        "procedurally generated levels of enemies. Reach level 5 to win."
    )

    # Frames advance only when an action is received (one action per beat).
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 70)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_EXIT = (255, 220, 0)
    COLOR_EXIT_GLOW = (255, 220, 0, 70)
    COLOR_BEAT_PULSE = (0, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 40, 60, 180)

    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Game Parameters
    MAX_HEALTH = 10
    BEATS_PER_LEVEL = 60
    MAX_LEVEL = 5
    MAX_EPISODE_STEPS = BEATS_PER_LEVEL * MAX_LEVEL

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.level = 1
        self.player_health = 0
        self.level_timer = 0
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.enemies = []
        self.particles = []
        self.last_player_dist_to_exit = 0.0

        # This will be properly initialized in reset()
        self.np_random = np.random.default_rng()

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.level = 1
        self.player_health = self.MAX_HEALTH
        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.level_timer -= 1
        self.steps += 1
        
        # --- Update Particles ---
        self._update_particles()

        # --- Player Action ---
        # 1. Movement
        prev_pos = self.player_pos
        px, py = self.player_pos
        if movement == 1 and py > 0: self.player_pos = (px, py - 1)
        elif movement == 2 and py < self.GRID_ROWS - 1: self.player_pos = (px, py + 1)
        elif movement == 3 and px > 0: self.player_pos = (px - 1, py)
        elif movement == 4 and px < self.GRID_COLS - 1: self.player_pos = (px + 1, py)

        # 2. Attack
        if space_pressed:
            reward += self._handle_attack()
        
        # --- Enemy Phase ---
        # Enemies attack if player moves next to them
        for enemy_pos in self.enemies:
            if self._manhattan_distance(self.player_pos, enemy_pos) == 1:
                self.player_health -= 1
                self._create_hit_effect(self.player_pos)
                # sfx: player_hit.wav

        # --- Reward Calculation ---
        current_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        if current_dist < self.last_player_dist_to_exit:
            reward += 1.0
        elif current_dist > self.last_player_dist_to_exit:
            reward -= 0.1
        self.last_player_dist_to_exit = current_dist

        # --- Check Level Completion ---
        if self.player_pos == self.exit_pos:
            if self.level >= self.MAX_LEVEL:
                reward += 500
                self.score += 1000
                self.game_over = True
                self.win_condition = True
                # sfx: game_win.wav
            else:
                reward += 100
                self.score += 200
                self.level += 1
                self._generate_level()
                # sfx: level_up.wav
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            # sfx: game_over.wav
        elif self.level_timer <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            # sfx: game_over_timeout.wav
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_beat_pulse()
        self._render_grid()
        self._render_exit()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "health": self.player_health,
            "level_timer": self.level_timer,
        }

    def _generate_level(self):
        self.level_timer = self.BEATS_PER_LEVEL
        self.enemies.clear()
        
        self.player_pos = (1, self.GRID_ROWS // 2)
        self.exit_pos = (self.GRID_COLS - 2, self.GRID_ROWS // 2)

        possible_spawns = []
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                pos = (x, y)
                if pos != self.player_pos and pos != self.exit_pos and self._manhattan_distance(pos, self.player_pos) > 2:
                    possible_spawns.append(pos)
        
        self.np_random.shuffle(possible_spawns)
        
        num_enemies = min(len(possible_spawns), self.level + 1)
        for i in range(num_enemies):
            self.enemies.append(possible_spawns[i])

        self.last_player_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

    def _handle_attack(self):
        reward = 0
        attacked_enemy = False
        px, py = self.player_pos
        
        # Check all 4 adjacent cells for enemies
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target_pos = (px + dx, py + dy)
            if target_pos in self.enemies:
                self.enemies.remove(target_pos)
                reward += 5
                self.score += 50
                attacked_enemy = True
                self._create_explosion(target_pos, self.COLOR_ENEMY)
                # sfx: enemy_die.wav

        if not attacked_enemy:
            # Penalty for whiffing an attack
            reward -= 1
            self.player_health -= 1
            self._create_hit_effect(self.player_pos, miss=True)
            # sfx: attack_miss.wav
        else:
            # sfx: attack_hit.wav
            pass

        return reward

    # --- Rendering Methods ---
    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_ROWS + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_beat_pulse(self):
        pulse_progress = (math.sin(self.steps * 0.4) + 1) / 2  # Oscillates between 0 and 1
        radius = int(self.GRID_WIDTH // 2 * (0.8 + pulse_progress * 0.2))
        alpha = int(30 + pulse_progress * 40)
        
        center_x = self.GRID_X_OFFSET + self.GRID_WIDTH // 2
        center_y = self.GRID_Y_OFFSET + self.GRID_HEIGHT // 2
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*self.COLOR_BEAT_PULSE, alpha))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*self.COLOR_BEAT_PULSE, alpha + 20))

    def _render_player(self):
        self._render_entity(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 0.8)

    def _render_enemies(self):
        for pos in self.enemies:
            self._render_entity(pos, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, 0.7)

    def _render_exit(self):
        self._render_entity(self.exit_pos, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, 0.7, shape='rect')

    def _render_entity(self, pos, color, glow_color, size_ratio, shape='circle'):
        cx = self.GRID_X_OFFSET + int((pos[0] + 0.5) * self.CELL_SIZE)
        cy = self.GRID_Y_OFFSET + int((pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * size_ratio / 2)
        
        # Glow effect
        glow_radius = int(radius * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, glow_radius, glow_color)
        
        # Main shape
        if shape == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
        elif shape == 'rect':
            rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.GRID_Y_OFFSET - 10)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        
        # Health
        health_text = self.font_small.render(f"HEALTH: {self.player_health}/{self.MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (20, 15))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (200, 15))
        
        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}/{self.MAX_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (380, 15))

        # Timer
        timer_text = self.font_small.render(f"BEATS LEFT: {self.level_timer}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (490, 15))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win_condition else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_EXIT if self.win_condition else self.COLOR_ENEMY)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text, text_rect)
        
        self.screen.blit(overlay, (0, 0))

    # --- Particle/Effect Methods ---
    def _create_explosion(self, grid_pos, color):
        cx = self.GRID_X_OFFSET + int((grid_pos[0] + 0.5) * self.CELL_SIZE)
        cy = self.GRID_Y_OFFSET + int((grid_pos[1] + 0.5) * self.CELL_SIZE)
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([cx, cy, vel[0], vel[1], lifetime, color])

    def _create_hit_effect(self, grid_pos, miss=False):
        cx = self.GRID_X_OFFSET + int((grid_pos[0] + 0.5) * self.CELL_SIZE)
        cy = self.GRID_Y_OFFSET + int((grid_pos[1] + 0.5) * self.CELL_SIZE)
        color = self.COLOR_GRID if miss else self.COLOR_ENEMY
        for i in range(4):
            angle = (math.pi / 2) * i + (math.pi / 4)
            speed = 4
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = 10
            self.particles.append([cx, cy, vel[0], vel[1], lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1

    def _render_particles(self):
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            size = max(0, int(2 * (lifetime / 20.0)))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

    # --- Utility Methods ---
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for interactive testing ---
    pygame.display.set_caption("Rhythm Roguelike")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    print("\n" + "="*30)
    print("      INTERACTIVE TEST MODE")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                
                # --- Map keys to actions for a single step ---
                # Movement
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                # Other keys
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                
                # If any key was pressed, take a step
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                    if terminated:
                        print("--- GAME OVER --- (Press R to reset)")


        # --- Render the observation from the environment ---
        # The environment's observation is (H, W, C), but pygame needs (W, H)
        # and surfarray.make_surface expects transposed array.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)  # Limit FPS for interactive mode

    env.close()