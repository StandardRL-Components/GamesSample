
# Generated: 2025-08-28T01:18:14.094406
# Source Brief: brief_04066.md
# Brief Index: 4066

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Catch all the bunnies before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must hunt down and catch five elusive bunnies on a grid before the 60-second timer expires."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

    MAX_BUNNIES_TO_CATCH = 5
    TIME_LIMIT_SECONDS = 60
    STEPS_PER_SECOND = 10
    MAX_STEPS = TIME_LIMIT_SECONDS * STEPS_PER_SECOND

    # Colors
    COLOR_BG = (28, 32, 40)
    COLOR_GRID = (40, 45, 55)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 120, 120)
    COLOR_BUNNY = (230, 230, 255)
    COLOR_BUNNY_EARS = (200, 200, 225)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_SUCCESS = (80, 255, 80)
    COLOR_FAIL = (255, 80, 80)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.bunny = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.bunnies_caught = 0
        self.bunny_escape_prob = 0.0
        self.game_over = False
        self.game_outcome = ""
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.bunnies_caught = 0
        self.game_over = False
        self.game_outcome = ""
        self.particles.clear()

        # Initial difficulty
        self.bunny_escape_prob = 0.4 # Starts at 40% chance to flee

        # Place player and first bunny
        self.player_pos = self._get_random_cell()
        self.bunny = self._spawn_bunny()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]
        # space_held and shift_held are not used in this game
        
        reward = -0.1  # Cost of time for encouraging efficiency
        self.steps += 1

        # --- Player Movement ---
        old_player_pos = self.player_pos
        dist_before = self._manhattan_distance(self.player_pos, self.bunny['pos'])
        
        if movement != 0: # 0 is no-op
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            new_x = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1] + dy))
            self.player_pos = (new_x, new_y)

        dist_after = self._manhattan_distance(self.player_pos, self.bunny['pos'])

        # Penalty for moving away from the bunny ("safe" move)
        if dist_after > dist_before:
            reward -= 2.0

        # --- Game Logic ---
        terminated = False
        
        # Check for bunny catch
        if self.player_pos == self.bunny['pos']:
            # SFX: Catch success
            self.score += 10
            reward += 10.0
            self.bunnies_caught += 1
            self._create_particles(self.player_pos, self.COLOR_SUCCESS, 20)
            
            if self.bunnies_caught >= self.MAX_BUNNIES_TO_CATCH:
                self.game_over = True
                terminated = True
                self.game_outcome = "VICTORY!"
                self.score += 50
                reward += 50.0
            else:
                self.bunny = self._spawn_bunny()
                # Increase difficulty
                self.bunny_escape_prob = min(1.0, 0.4 + self.bunnies_caught * 0.1)
        # Move bunny if player moved and game not over
        elif movement != 0:
            self._move_bunny()
            
        # Check for timeout
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
            self.game_outcome = "TIME OUT"
            self.score -= 50
            reward -= 50.0
            # SFX: Game over failure

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._update_and_draw_particles()
        if self.bunny:
            self._draw_bunny()
        self._draw_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bunnies_caught": self.bunnies_caught,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.STEPS_PER_SECOND,
        }

    # --- Helper and Rendering Methods ---

    def _get_random_cell(self):
        return (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))

    def _spawn_bunny(self):
        occupied_cells = [self.player_pos]
        pos = self._get_random_cell()
        while pos in occupied_cells:
            pos = self._get_random_cell()
        # SFX: Bunny spawn
        return {'pos': pos, 'bob': self.np_random.random() * 2 * math.pi}

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _move_bunny(self):
        bx, by = self.bunny['pos']
        px, py = self.player_pos
        
        possible_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = bx + dx, by + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) != self.player_pos:
                possible_moves.append((nx, ny))

        if not possible_moves:
            # Teleport if trapped
            self.bunny = self._spawn_bunny()
            return

        flee_move = None
        if self.np_random.random() < self.bunny_escape_prob:
            best_flee_move = None
            max_dist = self._manhattan_distance((bx, by), (px, py))
            
            # Find the move that maximizes distance from the player
            flee_candidates = []
            for move in possible_moves:
                dist = self._manhattan_distance(move, (px, py))
                if dist > max_dist:
                    max_dist = dist
                    flee_candidates = [move]
                elif dist == max_dist:
                    flee_candidates.append(move)
            
            if flee_candidates:
                flee_move = self.np_random.choice(flee_candidates)
                flee_move = tuple(flee_move)
        
        if flee_move:
            self.bunny['pos'] = flee_move
        else:
            # Random move from available options
            self.bunny['pos'] = self.np_random.choice(possible_moves)
            self.bunny['pos'] = tuple(self.bunny['pos'])

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_player(self):
        px, py = self.player_pos
        rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Glow effect
        glow_radius = int(self.CELL_SIZE * 0.8)
        glow_center = rect.center
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Main square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect.inflate(-4, -4))

    def _draw_bunny(self):
        self.bunny['bob'] = (self.bunny.get('bob', 0) + 0.3) % (2 * math.pi)
        bob_offset = int(math.sin(self.bunny['bob']) * 2)

        bx, by = self.bunny['pos']
        rect = pygame.Rect(bx * self.CELL_SIZE, by * self.CELL_SIZE - bob_offset, self.CELL_SIZE, self.CELL_SIZE)
        
        # Ears
        ear_width = self.CELL_SIZE // 4
        ear_height = self.CELL_SIZE // 2
        ear_left_rect = pygame.Rect(rect.left + ear_width // 2, rect.top - ear_height + 2, ear_width, ear_height)
        ear_right_rect = pygame.Rect(rect.right - ear_width * 1.5, rect.top - ear_height + 2, ear_width, ear_height)
        pygame.draw.rect(self.screen, self.COLOR_BUNNY_EARS, ear_left_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_BUNNY_EARS, ear_right_rect, border_radius=2)

        # Body
        body_rect = pygame.Rect(rect.left + 2, rect.top + 2, rect.width - 4, rect.height - 4)
        pygame.draw.rect(self.screen, self.COLOR_BUNNY, body_rect, border_radius=4)

    def _create_particles(self, grid_pos, color, count):
        px, py = (grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 2 + self.np_random.random() * 3
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = 20 + self.np_random.integers(0, 20)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 40))))
                size = int(self.CELL_SIZE * 0.2 * (p['life'] / 40))
                if size > 0:
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, (*p['color'], alpha), (size, size), size)
                    self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_text(self, text, font, position, color, shadow_color=None, align="topleft"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        
        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect(**{align: position})
            shadow_rect.topleft = (shadow_rect.left + 2, shadow_rect.top + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        setattr(text_rect, align, position)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_medium, (15, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Bunnies Caught
        self._render_text(f"Bunnies: {self.bunnies_caught} / {self.MAX_BUNNIES_TO_CATCH}", self.font_medium, (self.SCREEN_WIDTH // 2, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="midtop")
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.STEPS_PER_SECOND)
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_FAIL
        self._render_text(f"Time: {time_left:.1f}", self.font_medium, (self.SCREEN_WIDTH - 15, 10), time_color, self.COLOR_TEXT_SHADOW, align="topright")
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            outcome_color = self.COLOR_SUCCESS if self.game_outcome == "VICTORY!" else self.COLOR_FAIL
            self._render_text(self.game_outcome, self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), outcome_color, self.COLOR_TEXT_SHADOW, align="center")
            self._render_text(f"Final Score: {int(self.score)}", self.font_medium, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import sys
    
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + GameEnv.user_guide)
    print("Press 'R' to reset. Press 'ESC' to quit.")
    
    while running:
        action = [0, 0, 0]
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Game Reset ---")
                
                # Turn-based action on key press
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if action[0] != 0:
                    action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Bunnies: {info['bunnies_caught']}")

            if terminated:
                print(f"Game Over! {info}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()
    sys.exit()