
# Generated: 2025-08-28T06:25:59.132765
# Source Brief: brief_05902.md
# Brief Index: 5902

        
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
        "Controls: Arrow keys to move the target. Press space to squish bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Squish the descending bugs before they reach the bottom! Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = self.WIDTH // self.GRID_WIDTH

        # --- Colors ---
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TARGET = (0, 255, 128)
        self.COLOR_SPLAT = (120, 180, 80)
        self.BUG_COLORS = [
            (255, 80, 80),   # Red
            (80, 150, 255),  # Blue
            (255, 200, 80),  # Yellow
            (200, 100, 255)  # Purple
        ]
        
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = 0
        self.target_pos = [0, 0]
        self.bugs = []
        self.splats = []
        self.bug_speed = 0.0
        self.bug_spawn_rate = 0
        self.bug_spawn_timer = 0
        self.space_pressed_last_frame = False
        self.rng = np.random.default_rng()
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_STEPS
        
        self.target_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.bugs = []
        self.splats = []
        
        # Difficulty settings
        self.bug_speed = 0.0083  # 1 cell every ~2 seconds at 60fps
        self.bug_spawn_rate = self.FPS * 2  # Spawn every 2 seconds
        self.bug_spawn_timer = self.bug_spawn_rate
        
        self.space_pressed_last_frame = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0
        
        if not self.game_over:
            self.steps += 1
            self.timer -= 1
            
            # --- Handle Input ---
            self._handle_input(movement, space_held)
            
            # --- Game Logic ---
            self._update_bugs()
            self._update_splats()
            self._update_difficulty()
            self._spawn_bugs()
            
            # --- Calculate Reward ---
            squish_reward = self._check_squish(space_held)
            reward += squish_reward
            if squish_reward > 0:
                # Sound: squish.wav
                pass
            reward += 0.01 # Small survival reward per frame

            # --- Check Termination ---
            terminated, terminal_reward = self._check_termination()
            reward += terminal_reward
            if terminated:
                self.game_over = True
        
        else: # Game is over, just return current state
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.target_pos[1] -= 1
        if movement == 2: self.target_pos[1] += 1
        if movement == 3: self.target_pos[0] -= 1
        if movement == 4: self.target_pos[0] += 1
        
        self.target_pos[0] = np.clip(self.target_pos[0], 0, self.GRID_WIDTH - 1)
        self.target_pos[1] = np.clip(self.target_pos[1], 0, self.GRID_HEIGHT - 1)

    def _check_squish(self, space_held):
        reward = 0
        is_press_event = space_held and not self.space_pressed_last_frame
        self.space_pressed_last_frame = space_held
        
        if is_press_event:
            bugs_to_remove = []
            for bug in self.bugs:
                if int(bug['pos'][0]) == self.target_pos[0] and int(bug['pos'][1]) == self.target_pos[1]:
                    bugs_to_remove.append(bug)
            
            if bugs_to_remove:
                for bug in bugs_to_remove:
                    self.bugs.remove(bug)
                    self.score += 1
                    reward += 1
                    self._create_splat(bug['pos'], bug['color'])
        return reward
    
    def _create_splat(self, pos, color):
        splat_pos_px = (
            (pos[0] + 0.5) * self.CELL_SIZE,
            (pos[1] + 0.5) * self.CELL_SIZE
        )
        for _ in range(15):
            self.splats.append({
                'pos': list(splat_pos_px),
                'vel': [self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)],
                'lifetime': self.rng.integers(20, 40),
                'max_lifetime': 40,
                'color': color,
                'radius': self.rng.uniform(2, 5)
            })

    def _update_bugs(self):
        for bug in self.bugs:
            bug['pos'][1] += self.bug_speed

    def _update_splats(self):
        for splat in self.splats[:]:
            splat['pos'][0] += splat['vel'][0]
            splat['pos'][1] += splat['vel'][1]
            splat['vel'][0] *= 0.95
            splat['vel'][1] *= 0.95
            splat['lifetime'] -= 1
            if splat['lifetime'] <= 0:
                self.splats.remove(splat)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % (self.FPS) == 0: # Every second
            self.bug_speed += 0.0017
        if self.steps > 0 and self.steps % (self.FPS * 5) == 0: # Every 5 seconds
            self.bug_spawn_rate = max(self.FPS * 0.5, self.bug_spawn_rate - 5)

    def _spawn_bugs(self):
        self.bug_spawn_timer -= 1
        if self.bug_spawn_timer <= 0:
            self.bug_spawn_timer = self.rng.integers(int(self.bug_spawn_rate * 0.8), int(self.bug_spawn_rate * 1.2))
            
            x_pos = self.rng.integers(0, self.GRID_WIDTH)
            new_bug = {
                'pos': [x_pos, -0.5], # Start just off-screen
                'color': random.choice(self.BUG_COLORS),
                'leg_phase': self.rng.uniform(0, 2 * math.pi)
            }
            self.bugs.append(new_bug)
            # Sound: spawn.wav

    def _check_termination(self):
        # Loss condition: Bug reaches bottom
        for bug in self.bugs:
            if bug['pos'][1] >= self.GRID_HEIGHT:
                # Sound: fail.wav
                return True, -100 # Terminated, penalty

        # Win condition: Timer runs out
        if self.timer <= 0:
            self.win = True
            # Sound: win.wav
            return True, 100 # Terminated, bonus
            
        return False, 0 # Not terminated

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
            
        # Draw splats
        for splat in self.splats:
            alpha = int(255 * (splat['lifetime'] / splat['max_lifetime']))
            color = (*splat['color'], alpha)
            temp_surf = pygame.Surface((splat['radius']*2, splat['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (splat['radius'], splat['radius']), splat['radius'])
            self.screen.blit(temp_surf, (int(splat['pos'][0] - splat['radius']), int(splat['pos'][1] - splat['radius'])))

        # Draw bugs
        for bug in self.bugs:
            px = (bug['pos'][0] + 0.5) * self.CELL_SIZE
            py = (bug['pos'][1] + 0.5) * self.CELL_SIZE
            
            # Body
            body_radius = self.CELL_SIZE * 0.35
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(body_radius), bug['color'])
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(body_radius), bug['color'])
            
            # Legs
            leg_anim = math.sin(self.steps * 0.5 + bug['leg_phase']) * 4
            for i in range(3):
                angle_offset = math.pi / 3 * i
                for side in [-1, 1]:
                    angle = math.pi / 4 * side + angle_offset / 2 * side
                    start_x = px + math.cos(angle) * body_radius * 0.8
                    start_y = py + math.sin(angle) * body_radius * 0.8
                    end_x = start_x + math.cos(angle + leg_anim * side * 0.1) * (self.CELL_SIZE * 0.2)
                    end_y = start_y + math.sin(angle + leg_anim * side * 0.1) * (self.CELL_SIZE * 0.2)
                    pygame.draw.line(self.screen, bug['color'], (int(start_x), int(start_y)), (int(end_x), int(end_y)), 2)

        # Draw target
        tx, ty = self.target_pos
        target_rect = pygame.Rect(tx * self.CELL_SIZE, ty * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_TARGET, 50))
        self.screen.blit(s, (target_rect.x, target_rect.y))
        
        pygame.draw.rect(self.screen, self.COLOR_TARGET, target_rect, 2)
        
        cx, cy = target_rect.center
        cs = self.CELL_SIZE // 4
        pygame.draw.line(self.screen, self.COLOR_TARGET, (cx - cs, cy), (cx + cs, cy), 1)
        pygame.draw.line(self.screen, self.COLOR_TARGET, (cx, cy - cs), (cx, cy + cs), 1)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        seconds = self.timer // self.FPS
        time_text = self.font_main.render(f"TIME: {seconds}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "win": self.win,
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squish")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # no-op, no-space, no-shift

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Movement (one-shot, not continuous hold)
        movement_action = 0 # none
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        # Buttons (continuous hold)
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Reset action for next frame ---
        # For a human player, we want to reset movement so it's not sticky
        action = [0, 0, 0]
        
        # Limit frame rate
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()