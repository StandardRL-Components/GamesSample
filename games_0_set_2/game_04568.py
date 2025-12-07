import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style game where the player squashes swarming bugs.

    The player controls a cursor to click on bugs that appear on the screen.
    The goal is to squash a target number of bugs before too many escape.
    The difficulty increases as more bugs are squashed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to squash a bug."
    )

    # Short, user-facing description of the game
    game_description = (
        "Squash the swarming bugs with your cursor before they escape the garden!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Game parameters
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 10
        self.MAX_STEPS = 1500 # 50 seconds at 30fps
        self.MAX_BUGS = 20
        self.CURSOR_SPEED = 20
        self.BUG_RADIUS = 10
        self.CURSOR_RADIUS = 15
        self.EDGE_BONUS_MARGIN = 30
        self.DIFFICULTY_INTERVAL = 10
        self.INITIAL_BUG_SPEED = 1.0
        self.BUG_SPEED_INCREMENT = 0.1
        self.SPLAT_LIFETIME = 15 # frames

        # Colors
        self.COLOR_BG = (30, 60, 30)
        self.COLOR_PLAY_AREA = (80, 120, 80)
        self.COLOR_BUG = (220, 50, 50)
        self.COLOR_BUG_OUTLINE = (150, 20, 20)
        self.COLOR_SPLAT = (255, 220, 0)
        self.COLOR_CURSOR = (200, 255, 255)
        self.COLOR_CURSOR_OUTLINE = (100, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        
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
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            # Fallback if default font is not found (e.g., in some minimal environments)
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.bugs_escaped = 0
        self.game_over = False
        self.cursor_pos = None
        self.bugs = []
        self.splats = []
        self.base_bug_speed = self.INITIAL_BUG_SPEED
        self.prev_space_held = False
        self.np_random = None

        # self.reset() is not called in __init__ as per Gymnasium standard practice
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.bugs_escaped = 0
        self.game_over = False
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.bugs = []
        self.splats = []
        
        self.base_bug_speed = self.INITIAL_BUG_SPEED
        self.prev_space_held = False
        
        # Spawn initial bugs
        for _ in range(5):
            self._spawn_bug()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, _ = action
        is_click = space_held and not self.prev_space_held
        self.prev_space_held = bool(space_held)
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        reward = self._update_state(is_click)
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.bugs_squashed >= self.WIN_CONDITION:
            reward += 100
            terminated = True
        elif self.bugs_escaped >= self.LOSS_CONDITION:
            reward -= 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        
        # Clamp cursor to screen boundaries
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _update_state(self, is_click):
        reward = 0
        
        # Process click
        if is_click:
            # Iterate backwards to allow safe removal
            for i in range(len(self.bugs) - 1, -1, -1):
                bug = self.bugs[i]
                dist = np.linalg.norm(self.cursor_pos - bug['pos'])
                if dist < self.BUG_RADIUS + self.CURSOR_RADIUS:
                    # --- SQUASH ---
                    self.splats.append({'pos': bug['pos'].copy(), 'lifetime': self.SPLAT_LIFETIME, 'radius': self.BUG_RADIUS * 1.5})
                    self.bugs.pop(i)
                    
                    self.bugs_squashed += 1
                    self.score += 1
                    reward += 1
                    
                    # Check for edge bonus
                    if (bug['pos'][0] < self.EDGE_BONUS_MARGIN or
                        bug['pos'][0] > self.WIDTH - self.EDGE_BONUS_MARGIN or
                        bug['pos'][1] < self.EDGE_BONUS_MARGIN or
                        bug['pos'][1] > self.HEIGHT - self.EDGE_BONUS_MARGIN):
                        reward += 5
                        self.score += 5
                    
                    # Update difficulty
                    if self.bugs_squashed > 0 and self.bugs_squashed % self.DIFFICULTY_INTERVAL == 0:
                        self.base_bug_speed += self.BUG_SPEED_INCREMENT
                    
                    break # Only squash one bug per click
        
        # Update bugs
        bugs_to_remove = []
        for i, bug in enumerate(self.bugs):
            bug['pos'] += bug['vel']
            
            is_inside = (0 < bug['pos'][0] < self.WIDTH and 0 < bug['pos'][1] < self.HEIGHT)

            if bug['on_screen']:
                # Bug was inside, check if it has now escaped
                if not is_inside:
                    bugs_to_remove.append(i)
                    self.bugs_escaped += 1
                    self.score -= 10  # Penalty for escape
            else:
                # Bug was outside, check if it has now entered
                if is_inside:
                    bug['on_screen'] = True
                else:
                    # Bug is still outside. Clean up if it moves too far away.
                    margin = self.BUG_RADIUS * 3
                    px, py = bug['pos']
                    if not (-margin < px < self.WIDTH + margin and -margin < py < self.HEIGHT + margin):
                        bugs_to_remove.append(i)

        for i in sorted(bugs_to_remove, reverse=True):
            self.bugs.pop(i)
            
        # Update splats (visual effect)
        self.splats = [s for s in self.splats if s['lifetime'] > 0]
        for splat in self.splats:
            splat['lifetime'] -= 1

        # Spawn new bugs
        if len(self.bugs) < self.MAX_BUGS and self.np_random.random() < 0.1:
            self._spawn_bug()
            
        # Continuous negative reward for existing bugs
        reward -= 0.01 * len(self.bugs)
        
        return reward

    def _spawn_bug(self):
        # Spawn on one of the four edges
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.BUG_RADIUS])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.BUG_RADIUS])
        elif edge == 2: # Left
            pos = np.array([-self.BUG_RADIUS, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + self.BUG_RADIUS, self.np_random.uniform(0, self.HEIGHT)])

        # Aim towards a random point inside the screen to ensure they move inwards
        target = np.array([
            self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8),
            self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        ])
        
        direction = target - pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            norm = 1 # Avoid division by zero
        
        speed = self.base_bug_speed + self.np_random.uniform(-0.2, 0.2)
        vel = (direction / norm) * max(0.5, speed)
        
        self.bugs.append({'pos': pos, 'vel': vel, 'on_screen': False})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw play area border for reference
        pygame.draw.rect(self.screen, self.COLOR_PLAY_AREA, (0, 0, self.WIDTH, self.HEIGHT), 5)
        
        # Render splats
        for splat in self.splats:
            alpha = int(255 * (splat['lifetime'] / self.SPLAT_LIFETIME))
            radius = int(splat['radius'] * (1 - (splat['lifetime'] / self.SPLAT_LIFETIME)**2))
            if radius > 0:
                # Use gfxdraw for anti-aliased shapes
                pygame.gfxdraw.filled_circle(self.screen, int(splat['pos'][0]), int(splat['pos'][1]), radius, self.COLOR_SPLAT + (alpha,))

        # Render bugs
        for bug in self.bugs:
            x, y = int(bug['pos'][0]), int(bug['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BUG_RADIUS, self.COLOR_BUG_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BUG_RADIUS, self.COLOR_BUG)
            
            # Draw little legs for flair
            angle = math.atan2(bug['vel'][1], bug['vel'][0])
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    leg_angle = angle + i * (math.pi / 4 + j * 0.2)
                    start_pos = (x + self.BUG_RADIUS * 0.7 * math.cos(angle + i * math.pi/2),
                                 y + self.BUG_RADIUS * 0.7 * math.sin(angle + i * math.pi/2))
                    end_pos = (start_pos[0] + 5 * math.cos(leg_angle),
                               start_pos[1] + 5 * math.sin(leg_angle))
                    pygame.draw.aaline(self.screen, self.COLOR_BUG_OUTLINE, start_pos, end_pos)

        # Render cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.gfxdraw.aacircle(self.screen, cx, cy, self.CURSOR_RADIUS, self.COLOR_CURSOR_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.CURSOR_RADIUS, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cx - 5, cy), (cx + 5, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cx, cy - 5), (cx, cy + 5), 2)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Escaped bugs display
        escaped_text = self.font_large.render(f"Escaped: {self.bugs_escaped}/{self.LOSS_CONDITION}", True, self.COLOR_TEXT)
        text_rect = escaped_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(escaped_text, text_rect)
        
        # Squashed bugs display
        squashed_text = self.font_small.render(f"Squashed: {self.bugs_squashed}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(squashed_text, (12, 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_squashed": self.bugs_squashed,
            "bugs_escaped": self.bugs_escaped,
            "bug_speed": self.base_bug_speed
        }
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # The environment is designed to be headless, but we can add a display for testing.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Bug Squasher")
        
        obs, info = env.reset(seed=42)
        done = False
        
        print(env.user_guide)
        
        while not done:
            # Action mapping from keyboard to MultiDiscrete
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            # The third action component is unused in this game
            action = [movement, space, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            env.clock.tick(env.FPS)
            
        print(f"Game Over. Final Info: {info}")

    finally:
        env.close()