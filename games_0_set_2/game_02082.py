
# Generated: 2025-08-27T19:13:11.370845
# Source Brief: brief_02082.md
# Brief Index: 2082

        
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
        "Controls: Arrow keys to move the cursor. Press space to squash the bug in the selected cell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade action. Squash the bugs before they escape the grid. Faster bugs are worth more points!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_DIM = 10
        self.GAME_AREA_SIZE = 360
        self.CELL_SIZE = self.GAME_AREA_SIZE // self.GRID_DIM
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GAME_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GAME_AREA_SIZE) // 2
        
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_SQUASHED = 50
        self.LOSE_CONDITION_ESCAPED = 25
        self.INITIAL_SPAWN_COOLDOWN = 60  # 2 seconds at 30fps
        
        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.BUG_TYPES = {
            'green': {'color': (50, 205, 50), 'speed': 5, 'score': 1},
            'yellow': {'color': (255, 215, 0), 'speed': 3, 'score': 2},
            'red': {'color': (220, 20, 60), 'speed': 2, 'score': 3}
        }
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.bugs = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.bugs_squashed = 0
        self.bugs_escaped = 0
        self.bug_spawn_timer = 0
        self.base_bug_speed_mod = 1.0
        
        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        
        self.bugs = []
        self.particles = []
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.bugs_squashed = 0
        self.bugs_escaped = 0
        
        self.bug_spawn_timer = self.INITIAL_SPAWN_COOLDOWN
        self.base_bug_speed_mod = 1.0
        
        # Initial bug spawn
        for _ in range(3):
            self._spawn_bug()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # --- Action Handling ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        # 1. Handle cursor movement
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1)
        
        if movement == 0: # No-op
            reward -= 0.1
        elif self.cursor_pos != prev_cursor_pos: # Moved
            reward -= 0.1

        # 2. Handle squash action
        if space_pressed:
            bug_squashed = self._handle_squash()
            if not bug_squashed:
                # Optional: penalty for missing
                pass
            else:
                # Reward is handled in _handle_squash
                reward += bug_squashed['reward']
        
        # --- Game Logic Update ---
        self._update_bugs()
        self._update_particles()
        self._update_spawner()
        
        # Check for escaped bugs
        escaped_reward = self._check_escapes()
        reward += escaped_reward
        
        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.bugs_squashed >= self.WIN_CONDITION_SQUASHED:
            reward += 100
            terminated = True
            self.game_over = True
            self.game_over_message = "YOU WIN!"
        elif self.bugs_escaped >= self.LOSE_CONDITION_ESCAPED:
            reward -= 100
            terminated = True
            self.game_over = True
            self.game_over_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME UP"
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_squash(self):
        bug_to_remove = None
        squashed_bug_info = None
        for bug in self.bugs:
            if bug['pos'] == self.cursor_pos:
                # Found a bug to squash
                bug_to_remove = bug
                break
        
        if bug_to_remove:
            # Sfx: Squish sound
            self.bugs.remove(bug_to_remove)
            self.bugs_squashed += 1
            
            # Add rewards
            type_info = self.BUG_TYPES[bug_to_remove['type']]
            current_reward = 10 + type_info['score']
            self.score += current_reward
            
            squashed_bug_info = {'reward': current_reward}

            # Create particles
            self._create_particles(bug_to_remove['pos'], type_info['color'])
            
            # Difficulty scaling
            if self.bugs_squashed > 0 and self.bugs_squashed % 25 == 0:
                self.base_bug_speed_mod += 0.05
                # Sfx: Level up sound
        
        return squashed_bug_info

    def _update_bugs(self):
        for bug in self.bugs:
            bug['cooldown'] -= 1
            if bug['cooldown'] <= 0:
                # Move bug
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(moves)
                moved = False
                for dx, dy in moves:
                    new_x, new_y = bug['pos'][0] + dx, bug['pos'][1] + dy
                    if 0 <= new_x < self.GRID_DIM and 0 <= new_y < self.GRID_DIM:
                        bug['pos'] = [new_x, new_y]
                        moved = True
                        break
                
                # Reset cooldown
                speed = self.BUG_TYPES[bug['type']]['speed']
                bug['cooldown'] = max(1, int(speed / self.base_bug_speed_mod))

    def _check_escapes(self):
        escaped_bugs = []
        reward = 0
        for bug in self.bugs:
            x, y = bug['pos']
            if x == 0 or x == self.GRID_DIM - 1 or y == 0 or y == self.GRID_DIM - 1:
                escaped_bugs.append(bug)
        
        for bug in escaped_bugs:
            # Sfx: Escape sound
            self.bugs.remove(bug)
            self.bugs_escaped += 1
            reward -= 5
            self.score -= 5
        
        return reward

    def _update_spawner(self):
        self.bug_spawn_timer -= 1
        if self.bug_spawn_timer <= 0:
            self._spawn_bug()
            spawn_rate_decrease = self.steps / 200 # Gets faster over time
            self.bug_spawn_timer = max(10, self.INITIAL_SPAWN_COOLDOWN - spawn_rate_decrease)

    def _spawn_bug(self):
        # Spawn on an edge
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = [random.randint(0, self.GRID_DIM - 1), 0]
        elif edge == 'bottom':
            pos = [random.randint(0, self.GRID_DIM - 1), self.GRID_DIM - 1]
        elif edge == 'left':
            pos = [0, random.randint(0, self.GRID_DIM - 1)]
        else: # right
            pos = [self.GRID_DIM - 1, random.randint(0, self.GRID_DIM - 1)]
        
        # Ensure spawn position is not occupied
        is_occupied = any(bug['pos'] == pos for bug in self.bugs)
        if is_occupied:
            return # Try again next frame

        bug_type = random.choices(list(self.BUG_TYPES.keys()), weights=[0.5, 0.3, 0.2], k=1)[0]
        speed = self.BUG_TYPES[bug_type]['speed']
        
        self.bugs.append({
            'pos': pos,
            'type': bug_type,
            'cooldown': int(speed / self.base_bug_speed_mod),
        })

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        px += self.CELL_SIZE // 2
        py += self.CELL_SIZE // 2
        
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifetime'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_squashed": self.bugs_squashed,
            "bugs_escaped": self.bugs_escaped,
        }

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return x, y

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_DIM + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GAME_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GAME_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 20))
            color = p['color']
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), 
                max(0, int(p['lifetime'] / 4)), 
                (color[0], color[1], color[2], alpha)
            )

        # Draw bugs
        for bug in self.bugs:
            px, py = self._grid_to_pixel(bug['pos'])
            center = (px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2)
            radius = self.CELL_SIZE // 3
            color = self.BUG_TYPES[bug['type']]['color']
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (0,0,0), center, radius, 2) # Outline

        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos)
        cursor_rect = pygame.Rect(cx, cy, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Escaped
        escaped_text = self.font_small.render(f"ESCAPED: {self.bugs_escaped}/{self.LOSE_CONDITION_ESCAPED}", True, self.COLOR_TEXT)
        self.screen.blit(escaped_text, (self.SCREEN_WIDTH - escaped_text.get_width() - 10, 10))

        # Squashed
        squashed_text = self.font_small.render(f"SQUASHED: {self.bugs_squashed}/{self.WIN_CONDITION_SQUASHED}", True, self.COLOR_TEXT)
        self.screen.blit(squashed_text, (10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1
            
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()