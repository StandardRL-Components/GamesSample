
# Generated: 2025-08-27T17:58:24.620478
# Source Brief: brief_01694.md
# Brief Index: 1694

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Press Space to swap places with an adjacent crystal "
        "in your last direction of movement."
    )

    game_description = (
        "A strategic puzzle game. Collect 20 crystals within 50 moves to win. "
        "You can move onto crystals to collect them, or swap places with them to reposition."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE

        self.MAX_MOVES = 50
        self.CRYSTALS_TO_WIN = 20
        self.TOTAL_CRYSTALS = 25

        # --- Colors ---
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 220, 0)
        self.COLOR_CRYSTAL = (0, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (100, 255, 255)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 70)


        # --- Game State ---
        self.player_pos = None
        self.crystals = None
        self.moves_left = None
        self.crystals_collected = None
        self.score = None
        self.game_over = None
        self.win = None
        self.last_move_direction = None
        self.particles = []
        self.steps = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.crystals_collected = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        # Default direction for swapping on the first move
        self.last_move_direction = pygame.Vector2(0, -1) 

        # Place player in the center
        self.player_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        # Generate crystal positions
        self.crystals = []
        possible_positions = [pygame.Vector2(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_positions.remove(self.player_pos)
        
        # Use the environment's RNG for reproducibility
        shuffled_indices = self.np_random.permutation(len(possible_positions))
        shuffled_positions = [possible_positions[i] for i in shuffled_indices]
        
        self.crystals = shuffled_positions[:self.TOTAL_CRYSTALS]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        turn_taken = False

        dist_before = self._find_nearest_crystal_dist(self.player_pos)

        # --- Action Handling: Prioritize Swap over Move ---
        if space_held:
            # --- Swap Action ---
            turn_taken = True
            target_swap_pos = self.player_pos + self.last_move_direction
            
            crystal_to_swap_idx = -1
            for i, crystal_pos in enumerate(self.crystals):
                if crystal_pos == target_swap_pos:
                    crystal_to_swap_idx = i
                    break
            
            if crystal_to_swap_idx != -1:
                # Perform the swap
                # Sound: Swap sfx
                self.crystals[crystal_to_swap_idx] = self.player_pos.copy()
                self.player_pos = target_swap_pos.copy()

        elif movement != 0:
            # --- Movement Action ---
            turn_taken = True
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y = -1  # Up
            elif movement == 2: move_vec.y = 1   # Down
            elif movement == 3: move_vec.x = -1  # Left
            elif movement == 4: move_vec.x = 1   # Right
            
            self.last_move_direction = move_vec
            new_pos = self.player_pos + move_vec

            # Check boundaries
            if 0 <= new_pos.x < self.GRID_WIDTH and 0 <= new_pos.y < self.GRID_HEIGHT:
                self.player_pos = new_pos

        # --- Post-Action Logic ---
        if turn_taken:
            self.moves_left -= 1

            # Check for crystal collection
            if self.player_pos in self.crystals:
                # Sound: Crystal collect
                self.crystals.remove(self.player_pos)
                self.crystals_collected += 1
                reward += 10
                self._spawn_particles(self.player_pos, self.COLOR_PARTICLE)

            # Proximity reward
            dist_after = self._find_nearest_crystal_dist(self.player_pos)
            if dist_after < dist_before:
                reward += 1.0
            elif dist_after > dist_before:
                reward -= 0.1
        
        # --- Update game state ---
        self.steps += 1
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.crystals_collected >= self.CRYSTALS_TO_WIN:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw crystals
        for crystal_pos in self.crystals:
            center_x = int(crystal_pos.x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(crystal_pos.y * self.CELL_SIZE + self.CELL_SIZE / 2)
            s = self.CELL_SIZE * 0.35
            points = [(center_x, center_y - s), (center_x + s, center_y), (center_x, center_y + s), (center_x - s, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

            # Sparkle effect
            sparkle_alpha = (math.sin(pygame.time.get_ticks() * 0.005 + crystal_pos.x) + 1) / 2 * 200 + 55
            sparkle_color = (255, 255, 255, int(sparkle_alpha))
            sparkle_size = self.CELL_SIZE * 0.1
            pygame.draw.line(self.screen, sparkle_color, (center_x - sparkle_size, center_y), (center_x + sparkle_size, center_y), 2)
            pygame.draw.line(self.screen, sparkle_color, (center_x, center_y - sparkle_size), (center_x, center_y + sparkle_size), 2)

        # Draw particles
        for p in self.particles:
            p_pos = p['pos'] * self.CELL_SIZE + pygame.Vector2(self.CELL_SIZE/2, self.CELL_SIZE/2)
            alpha = max(0, p['life'] / p['max_life'])
            color = (*p['color'], int(alpha * 255))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p_pos.x), int(p_pos.y), radius, color)

        # Draw player
        player_rect = pygame.Rect(
            self.player_pos.x * self.CELL_SIZE + 5,
            self.player_pos.y * self.CELL_SIZE + 5,
            self.CELL_SIZE - 10,
            self.CELL_SIZE - 10,
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), player_rect, width=2, border_radius=4)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Crystals Collected
        crystals_text = self.font_main.render(f"Crystals: {self.crystals_collected}/{self.CRYSTALS_TO_WIN}", True, self.COLOR_TEXT)
        text_rect = crystals_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(crystals_text, text_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "crystals_collected": self.crystals_collected,
        }

    def _find_nearest_crystal_dist(self, pos):
        if not self.crystals:
            return 0
        
        min_dist = float('inf')
        for crystal_pos in self.crystals:
            dist = pos.distance_to(crystal_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _spawn_particles(self, grid_pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.05, 0.15)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': grid_pos.copy(),
                'vel': vel,
                'radius': self.np_random.integers(3, 7),
                'life': life,
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11" or "windows" or "dummy"

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Grid")
    clock = pygame.time.Clock()
    
    running = True
    action_taken_this_frame = False

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken_this_frame = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
        
        if not env.game_over and action_taken_this_frame:
            obs, reward, done, _, info = env.step(action)
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
        
        action_taken_this_frame = False

        frame = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()