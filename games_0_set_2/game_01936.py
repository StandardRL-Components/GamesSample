import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to click."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle. Click the bugs as they appear to score points "
        "before the timer runs out. Clear 20 bugs to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.WIN_BUGS_CLEARED = 20
        self.CURSOR_SPEED = 10
        self.BUG_RADIUS = 12
        self.BUG_PROXIMITY_RADIUS = self.BUG_RADIUS + 10

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_OUTLINE = (200, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TIMER_START = (0, 220, 120)
        self.COLOR_TIMER_END = (255, 70, 70)
        self.BUG_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 150, 255), # Blue
            (255, 255, 80), # Yellow
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.cursor_pos = None
        self.bugs = None
        self.particles = None
        self.steps = None
        self.score = None
        self.bugs_cleared = None
        self.time_remaining = None
        self.game_over = None
        self.win = None
        self.last_space_state = None
        self.bug_spawn_timer = None
        self.bugs_per_second = None
        self.current_spawn_delay = None
        
        self.rng = np.random.default_rng()
        
        # self.reset() # This is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is for debugging, not needed in final version.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.bugs = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.bugs_cleared = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win = False
        self.last_space_state = False

        # Difficulty and spawning
        self.bugs_per_second = 0.5 # Initial spawn rate: 1 bug / 2 seconds
        self.current_spawn_delay = self.FPS / self.bugs_per_second
        self.bug_spawn_timer = self.current_spawn_delay

        # Spawn initial bugs
        for _ in range(3):
            self._spawn_bug()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Game Logic Update ---
        self._handle_input(movement, space_held)
        
        # Click event handling
        click_event = space_held and not self.last_space_state
        if click_event:
            reward += self._handle_click()
        self.last_space_state = space_held

        # Update timers and spawning
        self.time_remaining -= 1
        self._update_spawning()
        
        # Update particles
        self._update_particles()
        
        # Increment step counter
        self.steps += 1
        
        # Check for termination
        terminated = False
        if self.bugs_cleared >= self.WIN_BUGS_CLEARED:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100  # Goal-oriented reward for winning
        elif self.time_remaining <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 50  # Penalty for running out of time
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        if movement == 1: # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
            
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
    def _handle_click(self):
        click_reward = 0
        hit_bug = False

        # Check for direct hits
        for i in range(len(self.bugs) - 1, -1, -1):
            bug = self.bugs[i]
            dist = np.linalg.norm(self.cursor_pos - bug['pos'])
            if dist < bug['radius']:
                self._create_particles(bug['pos'], bug['color'], 20, 4)
                self.bugs.pop(i)
                
                self.score += 10
                self.bugs_cleared += 1
                click_reward += 10  # Event-based reward for clearing a bug
                hit_bug = True
                break # Only one bug per click

        # If no bug was hit, check for proximity or miss
        if not hit_bug:
            self._create_particles(self.cursor_pos, (100, 100, 120), 5, 1.5, life=15)
            
            is_near_bug = False
            for bug in self.bugs:
                dist = np.linalg.norm(self.cursor_pos - bug['pos'])
                if dist < self.BUG_PROXIMITY_RADIUS:
                    click_reward += 1 # Continuous feedback for being close
                    is_near_bug = True
                    break
            
            if not is_near_bug:
                click_reward -= 0.1 # Penalty for clicking empty space
                
        return click_reward

    def _update_spawning(self):
        # Difficulty scaling: Increase spawn rate every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.bugs_per_second += 0.1
            self.current_spawn_delay = max(15, self.FPS / self.bugs_per_second) # Cap at 4 bugs/sec

        # Spawn new bug if timer is up
        self.bug_spawn_timer -= 1
        if self.bug_spawn_timer <= 0 and not self.game_over:
            self._spawn_bug()
            self.bug_spawn_timer = self.current_spawn_delay

    def _spawn_bug(self):
        # SFX: Bug Spawn
        low = [self.BUG_RADIUS, self.BUG_RADIUS]
        high = [self.SCREEN_WIDTH - self.BUG_RADIUS, self.SCREEN_HEIGHT - self.BUG_RADIUS]
        pos = self.rng.uniform(low, high, size=2).astype(np.float32)

        color = self.BUG_COLORS[self.rng.integers(len(self.BUG_COLORS))]
        pulse_offset = self.rng.random() * 2 * math.pi
        self.bugs.append({'pos': pos, 'color': color, 'radius': self.BUG_RADIUS, 'pulse_offset': pulse_offset})
        self._create_particles(pos, color, 10, 2, life=20)

    def _create_particles(self, pos, color, count, speed_scale, life=30):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = (self.rng.random() + 0.1) * speed_scale
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life + self.rng.integers(-5, 5),
                'color': color
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw particles
        for p in self.particles:
            size = max(0, int(p['life'] / 6))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), size)

        # Draw bugs
        for bug in self.bugs:
            pulse = (math.sin(self.steps * 0.15 + bug['pulse_offset']) + 1) / 2
            radius = int(bug['radius'] + pulse * 3)
            pos_int = bug['pos'].astype(int)
            
            # Glow effect
            glow_radius = int(radius * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            glow_color = bug['color'] + (int(80 + pulse * 40),)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, bug['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, bug['color'])
            
        # Draw cursor
        cursor_pos_int = self.cursor_pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, cursor_pos_int, 8, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0] - 12, cursor_pos_int[1]), (cursor_pos_int[0] - 4, cursor_pos_int[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0] + 4, cursor_pos_int[1]), (cursor_pos_int[0] + 12, cursor_pos_int[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0], cursor_pos_int[1] - 12), (cursor_pos_int[0], cursor_pos_int[1] - 4), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0], cursor_pos_int[1] + 4), (cursor_pos_int[0], cursor_pos_int[1] + 12), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Bugs Cleared
        bugs_text = self.font_main.render(f"BUGS: {self.bugs_cleared}/{self.WIN_BUGS_CLEARED}", True, self.COLOR_TEXT)
        self.screen.blit(bugs_text, (self.SCREEN_WIDTH - bugs_text.get_width() - 10, 10))

        # Timer bar
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        timer_width = int(time_ratio * self.SCREEN_WIDTH)
        timer_color = [
            int(s + (e - s) * (1 - time_ratio)) 
            for s, e in zip(self.COLOR_TIMER_START, self.COLOR_TIMER_END)
        ]
        pygame.draw.rect(self.screen, timer_color, (0, 0, timer_width, 5))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_TIMER_START)
            else:
                end_text = self.font_large.render("TIME'S UP!", True, self.COLOR_TIMER_END)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_cleared": self.bugs_cleared,
            "time_remaining_seconds": self.time_remaining / self.FPS,
        }
        
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