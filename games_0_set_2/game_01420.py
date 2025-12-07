
# Generated: 2025-08-27T17:05:32.871043
# Source Brief: brief_01420.md
# Brief Index: 1420

        
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
    """
    An arcade racing/puzzle game where the player guides a snail to a finish line.

    The player controls the snail on a grid, aiming to reach the rightmost edge.
    Each move consumes a limited resource ("moves"). Obstacles (rocks) consume
    extra moves, while boosts provide a score bonus. The game rewards finishing
    quickly with many moves remaining. The difficulty increases after each successful
    run by adding more obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the snail. Reach the checkered finish line on the right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a snail through a race track. Avoid rocks and use boosts to finish with the most moves left."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 50
    CELL_SIZE = 8
    GRID_PIXEL_WIDTH = GRID_SIZE * CELL_SIZE  # 400
    GRID_PIXEL_HEIGHT = GRID_SIZE * CELL_SIZE # 400
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2

    # Colors
    COLOR_BG = (25, 45, 25)
    COLOR_GRID = (40, 60, 40)
    COLOR_FINISH_1 = (200, 200, 200)
    COLOR_FINISH_2 = (150, 150, 150)
    
    COLOR_SNAIL = (255, 80, 80)
    COLOR_SNAIL_EYE = (255, 255, 255)
    COLOR_SNAIL_PUPIL = (0, 0, 0)
    COLOR_SNAIL_TRAIL = (255, 120, 120, 150)

    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_OBSTACLE_HIT = (180, 100, 50)
    COLOR_BOOST = (50, 255, 50)
    COLOR_BOOST_SPARKLE = (200, 255, 200)

    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_WIN_TEXT = (100, 255, 100)
    COLOR_LOSE_TEXT = (255, 100, 100)
    COLOR_OVERLAY = (0, 0, 0, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # State variables initialized in reset
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_outcome = None
        self.max_moves = 100
        self.remaining_moves = None
        self.elapsed_time = None
        self.start_pos = None
        self.snail_pos = None
        self.last_move_dir = np.array([1, 0])
        self.finish_pos_x = self.GRID_SIZE - 1
        self.obstacles = None
        self.boosts = None
        self.particles = None
        
        # Difficulty progression
        self.num_obstacles_start = 5
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_outcome = None
        self.remaining_moves = self.max_moves
        self.elapsed_time = 0.0
        
        self.start_pos = [2, self.GRID_SIZE // 2]
        self.snail_pos = self.start_pos[:]
        self.last_move_dir = np.array([1, 0])

        self.obstacles = set()
        self.boosts = set()
        self.particles = []
        self._place_objects()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Cost of existing per step

        moved = False
        
        if movement == 1: # Up
            self.snail_pos[1] -= 1
            self.last_move_dir = np.array([0, -1])
            moved = True
        elif movement == 2: # Down
            self.snail_pos[1] += 1
            self.last_move_dir = np.array([0, 1])
            moved = True
        elif movement == 3: # Left
            self.snail_pos[0] -= 1
            self.last_move_dir = np.array([-1, 0])
            moved = True
        elif movement == 4: # Right
            self.snail_pos[0] += 1
            self.last_move_dir = np.array([1, 0])
            moved = True

        # Clamp position to grid
        self.snail_pos[0] = np.clip(self.snail_pos[0], 0, self.GRID_SIZE - 1)
        self.snail_pos[1] = np.clip(self.snail_pos[1], 0, self.GRID_SIZE - 1)

        if moved:
            self.remaining_moves -= 1
            self.elapsed_time += 0.5 # Fixed time per move
            self._add_particles(self.snail_pos, 3, self.COLOR_SNAIL_TRAIL, 'trail')

            # Check for obstacle collision
            if tuple(self.snail_pos) in self.obstacles:
                reward -= 2.0
                self.remaining_moves -= 5
                self.obstacles.remove(tuple(self.snail_pos))
                self._add_particles(self.snail_pos, 15, self.COLOR_OBSTACLE_HIT, 'hit')
                # SFX: Rock crumble

            # Check for boost collision
            if tuple(self.snail_pos) in self.boosts:
                reward += 5.0
                self.boosts.remove(tuple(self.snail_pos))
                self._add_particles(self.snail_pos, 20, self.COLOR_BOOST_SPARKLE, 'boost')
                # SFX: Powerup get
        
        self._update_particles()
        self.score += reward
        terminated = False

        # Check for win condition
        if self.snail_pos[0] >= self.finish_pos_x:
            win_bonus = 100 * (max(0, self.remaining_moves) / self.max_moves)
            reward += win_bonus
            self.score += win_bonus
            terminated = True
            self.game_over = True
            self.game_outcome = "YOU WIN!"
            self.num_obstacles_start = min(200, self.num_obstacles_start + 2)
            # SFX: Win jingle

        # Check for loss condition
        if self.remaining_moves <= 0 and not terminated:
            reward -= 10.0 # Extra penalty for running out of moves
            self.score -= 10.0
            terminated = True
            self.game_over = True
            self.game_outcome = "OUT OF MOVES"
            # SFX: Lose sound
            
        self.steps += 1
        if self.steps >= 1000 and not terminated:
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME OUT"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_objects(self):
        # Create a set of forbidden positions (start area, finish line)
        forbidden_pos = set()
        for i in range(5):
            for j in range(self.GRID_SIZE):
                forbidden_pos.add((i, j)) # Start area
                forbidden_pos.add((self.finish_pos_x, j)) # Finish line
        
        # Place obstacles
        for _ in range(self.num_obstacles_start):
            for _ in range(100): # Attempt 100 times to find a free spot
                pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
                if pos not in forbidden_pos:
                    self.obstacles.add(pos)
                    forbidden_pos.add(pos)
                    break
        
        # Place boosts
        num_boosts = 5
        for _ in range(num_boosts):
            for _ in range(100): # Attempt 100 times
                pos = tuple(self.np_random.integers(0, self.GRID_SIZE, size=2))
                if pos not in forbidden_pos:
                    self.boosts.add(pos)
                    forbidden_pos.add(pos)
                    break

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.98

    def _add_particles(self, grid_pos, count, color, p_type):
        center_px = self._grid_to_pixel_center(grid_pos[0], grid_pos[1])
        for _ in range(count):
            if p_type == 'trail':
                vel = [0, 0]
                lifetime = 20
                radius = self.np_random.uniform(1, 3)
            elif p_type == 'hit':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifetime = 15
                radius = self.np_random.uniform(2, 4)
            elif p_type == 'boost':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifetime = 25
                radius = self.np_random.uniform(2, 5)

            self.particles.append({
                'pos': list(center_px), 'vel': vel, 'radius': radius,
                'lifetime': lifetime, 'max_lifetime': lifetime, 'color': color
            })
    
    def _grid_to_pixel(self, gx, gy):
        return (
            self.GRID_OFFSET_X + gx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + gy * self.CELL_SIZE
        )

    def _grid_to_pixel_center(self, gx, gy):
        return (
            self.GRID_OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_PIXEL_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, y))

        # Draw finish line
        for y in range(self.GRID_SIZE):
            color = self.COLOR_FINISH_1 if (y + (self.finish_pos_x % 2)) % 2 == 0 else self.COLOR_FINISH_2
            rect = pygame.Rect(self._grid_to_pixel(self.finish_pos_x, y), (self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, color, rect)

    def _render_objects(self):
        # Draw obstacles
        for obs_pos in self.obstacles:
            px, py = self._grid_to_pixel_center(obs_pos[0], obs_pos[1])
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), self.CELL_SIZE // 2, self.COLOR_OBSTACLE)
        
        # Draw boosts
        flash_val = math.sin(self.steps * 0.3)
        for boost_pos in self.boosts:
            px, py = self._grid_to_pixel_center(boost_pos[0], boost_pos[1])
            radius = int(self.CELL_SIZE / 2.5 + flash_val * 1.5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_BOOST)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_BOOST)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = p['color']
            
            # Create a temporary surface for alpha blending
            radius = int(p['radius'])
            if radius > 0:
                surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color[:3], alpha), (radius, radius), radius)
                self.screen.blit(surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_snail(self):
        px, py = self._grid_to_pixel_center(self.snail_pos[0], self.snail_pos[1])
        radius = self.CELL_SIZE // 2 + 1
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_SNAIL)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_SNAIL)
        
        # Eyes
        eye_offset_mag = radius * 0.6
        eye_base_1 = (px + self.last_move_dir[1] * eye_offset_mag, py - self.last_move_dir[0] * eye_offset_mag)
        eye_base_2 = (px - self.last_move_dir[1] * eye_offset_mag, py + self.last_move_dir[0] * eye_offset_mag)
        
        eye_look = (self.last_move_dir[0] * 2, self.last_move_dir[1] * 2)

        for base in [eye_base_1, eye_base_2]:
            pygame.gfxdraw.filled_circle(self.screen, int(base[0]), int(base[1]), 2, self.COLOR_SNAIL_EYE)
            pygame.gfxdraw.filled_circle(self.screen, int(base[0] + eye_look[0]), int(base[1] + eye_look[1]), 1, self.COLOR_SNAIL_PUPIL)

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {max(0, self.remaining_moves)}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (10, 10))

        # Time elapsed
        time_text = f"Time: {self.elapsed_time:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

    def _render_game_over(self):
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            color = self.COLOR_WIN_TEXT if "WIN" in self.game_outcome else self.COLOR_LOSE_TEXT
            text_surf = self.font_game_over.render(self.game_outcome, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self._render_background()
        self._render_objects()
        self._render_particles()
        self._render_snail()
        self._render_ui()
        self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "level": self.num_obstacles_start
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print(env.game_description)
    print(env.user_guide)
    print("\nPress 'R' to reset, 'Q' to quit.")

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Snail Racer")
    clock = pygame.time.Clock()

    game_is_over = False
    while not done:
        action = [0, 0, 0] # Default: no-op
        human_action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if game_is_over: # If game is over, any key press resets
                    obs, info = env.reset()
                    game_is_over = False
                    continue

                human_action_taken = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    human_action_taken = False
                elif event.key == pygame.K_q:
                    done = True
                    human_action_taken = False
                else: # Any other key is a no-op
                    human_action_taken = False
        
        if human_action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['remaining_moves']}")
            if terminated:
                game_is_over = True
                print(f"Game Over! Final Score: {info['score']:.2f}. Press any key to play again.")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    pygame.quit()