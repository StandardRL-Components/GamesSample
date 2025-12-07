
# Generated: 2025-08-27T21:53:07.665485
# Source Brief: brief_02942.md
# Brief Index: 2942

        
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
    An arcade-style game where the player hunts colorful bugs on a grid.

    The player's goal is to catch 15 bugs before 5 of them escape by
    reaching the edge of the grid. The game is turn-based, with the world
    state advancing only when the player submits an action.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use Arrow Keys (↑, ↓, ←, →) to move your avatar one square at a time. "
        "Catch bugs by moving onto their square."
    )
    game_description = (
        "Hunt colorful bugs in a grid-based arena. Catch 15 bugs to win, "
        "but let 5 escape and you lose. Bug speed increases as you catch more!"
    )

    # Frame advance setting
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40

    # Colors
    COLOR_BG_TOP = (15, 20, 30)
    COLOR_BG_BOTTOM = (30, 40, 60)
    COLOR_GRID = (50, 60, 80)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    BUG_COLORS = {
        "red": (255, 80, 80),
        "blue": (80, 150, 255),
        "yellow": (255, 255, 80)
    }
    COLOR_TEXT = (220, 220, 240)
    COLOR_ESCAPE_FLASH = (200, 0, 0)

    # Game Parameters
    WIN_CONDITION_BUGS_CAUGHT = 15
    LOSE_CONDITION_BUGS_ESCAPED = 5
    MAX_STEPS = 1000
    INITIAL_BUG_COUNT = 7
    DIFFICULTY_INTERVAL = 5 # Increase speed every 5 bugs caught
    DIFFICULTY_SPEED_INCREASE = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Internal state variables (initialized in reset)
        self.player_pos = None
        self.bugs = None
        self.particles = None
        self.steps = None
        self.score = None
        self.bugs_caught = None
        self.bugs_escaped = None
        self.bug_speed_modifier = None
        self.escape_flash_timer = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.bugs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.bugs_caught = 0
        self.bugs_escaped = 0
        self.bug_speed_modifier = 1.0
        self.escape_flash_timer = 0
        
        for _ in range(self.INITIAL_BUG_COUNT):
            self._spawn_bug()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held and shift_held are unused in this game
        
        reward = 0
        terminated = False
        
        # --- 1. Player Movement and Reward Calculation ---
        old_pos = list(self.player_pos)
        dist_before = self._get_distance_to_nearest_bug(old_pos)
        
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0]))
        self.player_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1]))
        
        dist_after = self._get_distance_to_nearest_bug(self.player_pos)
        
        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1

        # --- 2. Bug Catching ---
        bugs_to_remove = []
        for bug in self.bugs:
            if bug['pos'] == self.player_pos:
                bugs_to_remove.append(bug)
                self.bugs_caught += 1
                self.score += bug['value']
                reward += 10
                self._create_particles(self.player_pos, bug['color'])
                # SFX: Play catch sound

        for bug in bugs_to_remove:
            self.bugs.remove(bug)
            self._spawn_bug()
        
        # --- 3. Bug Movement and Escapes ---
        bugs_to_remove = []
        for bug in self.bugs:
            # Determine number of moves based on speed modifier
            moves = 1
            if self.np_random.random() < (self.bug_speed_modifier - 1.0):
                moves = 2

            for _ in range(moves):
                if bug in bugs_to_remove: continue # Already escaped this turn
                
                move_dir = self.np_random.integers(0, 4) # 0:U, 1:D, 2:L, 3:R
                if move_dir == 0: bug['pos'][1] -= 1
                elif move_dir == 1: bug['pos'][1] += 1
                elif move_dir == 2: bug['pos'][0] -= 1
                elif move_dir == 3: bug['pos'][0] += 1

                if not (0 <= bug['pos'][0] < self.GRID_WIDTH and 0 <= bug['pos'][1] < self.GRID_HEIGHT):
                    bugs_to_remove.append(bug)
                    self.bugs_escaped += 1
                    reward -= 5
                    self.escape_flash_timer = 5 # Flash for 5 frames
                    # SFX: Play escape sound
                    break # Stop moving if escaped

        for bug in bugs_to_remove:
            if bug in self.bugs: # Check if not already caught in the same step
                self.bugs.remove(bug)
                self._spawn_bug()

        # --- 4. Update Game State ---
        self.steps += 1
        self._update_difficulty()
        self._update_particles()
        if self.escape_flash_timer > 0:
            self.escape_flash_timer -= 1

        # --- 5. Check Termination Conditions ---
        if self.bugs_caught >= self.WIN_CONDITION_BUGS_CAUGHT:
            terminated = True
            reward += 50
        elif self.bugs_escaped >= self.LOSE_CONDITION_BUGS_ESCAPED:
            terminated = True
            reward -= 50
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_grid()
        self._render_bugs()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        if self.escape_flash_timer > 0:
            self._render_escape_flash()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_caught": self.bugs_caught,
            "bugs_escaped": self.bugs_escaped,
        }

    # --- Helper and Rendering Methods ---

    def _spawn_bug(self):
        """Spawns a new bug at a random location, not on the player."""
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if pos != self.player_pos and not any(bug['pos'] == pos for bug in self.bugs):
                break
        
        bug_type = self.np_random.choice(list(self.BUG_COLORS.keys()))
        self.bugs.append({
            "pos": pos,
            "color": self.BUG_COLORS[bug_type],
            "value": 10, # Could be different for different colors
            "anim_offset": self.np_random.random() * 2 * math.pi
        })

    def _get_distance_to_nearest_bug(self, pos):
        """Calculates Manhattan distance to the nearest bug."""
        if not self.bugs:
            return float('inf')
        return min(abs(pos[0] - b['pos'][0]) + abs(pos[1] - b['pos'][1]) for b in self.bugs)

    def _update_difficulty(self):
        """Increases bug speed based on the number of bugs caught."""
        self.bug_speed_modifier = 1.0 + (self.bugs_caught // self.DIFFICULTY_INTERVAL) * self.DIFFICULTY_SPEED_INCREASE

    def _render_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_grid(self):
        """Draws the grid lines."""
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_player(self):
        """Renders the player's avatar."""
        center_x = int((self.player_pos[0] + 0.5) * self.CELL_SIZE)
        center_y = int((self.player_pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.35)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_OUTLINE)

    def _render_bugs(self):
        """Renders all bugs with a breathing animation."""
        for bug in self.bugs:
            center_x = int((bug['pos'][0] + 0.5) * self.CELL_SIZE)
            center_y = int((bug['pos'][1] + 0.5) * self.CELL_SIZE)
            
            # Breathing animation
            anim_phase = (self.steps * 0.1 + bug['anim_offset'])
            radius_mod = 0.05 * math.sin(anim_phase)
            radius = int(self.CELL_SIZE * (0.25 + radius_mod))
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, bug['color'])
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, bug['color'])

    def _create_particles(self, grid_pos, color):
        """Creates a burst of particles for visual feedback."""
        center_x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        center_y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': 20, # Frames
            })

    def _update_particles(self):
        """Updates position and life of all particles."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        """Renders all active particles."""
        for p in self.particles:
            size = max(0, int(p['life'] * 0.2))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.rect(self.screen, p['color'], (*pos, size, size))

    def _render_ui(self):
        """Renders the score and other game info."""
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        caught_text = self.font_small.render(f"CAUGHT: {self.bugs_caught}/{self.WIN_CONDITION_BUGS_CAUGHT}", True, self.COLOR_TEXT)
        self.screen.blit(caught_text, (15, 50))
        
        escaped_text = self.font_small.render(f"ESCAPED: {self.bugs_escaped}/{self.LOSE_CONDITION_BUGS_ESCAPED}", True, self.COLOR_TEXT)
        self.screen.blit(escaped_text, (15, 75))

    def _render_escape_flash(self):
        """Renders a red border flash when a bug escapes."""
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        alpha = int(100 * (self.escape_flash_timer / 5.0)) # Fade out
        pygame.draw.rect(s, (*self.COLOR_ESCAPE_FLASH, alpha), s.get_rect(), 10)
        self.screen.blit(s, (0, 0))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Requires pygame to be installed with display drivers.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bug Hunter")

    done = False
    action = [0, 0, 0] # No-op, release space, release shift

    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                
                # Reset action on new key press
                action = [0, 0, 0]
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Since auto_advance is False, we step on key press
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Caught: {info['bugs_caught']}, Escaped: {info['bugs_escaped']}")
                if terminated:
                    print("\n--- GAME OVER ---")
                    if info['bugs_caught'] >= GameEnv.WIN_CONDITION_BUGS_CAUGHT:
                        print("YOU WIN!")
                    else:
                        print("YOU LOSE!")
                    print(f"Final Score: {info['score']}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0,0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only tick the clock to limit CPU usage
        env.clock.tick(60)

    env.close()