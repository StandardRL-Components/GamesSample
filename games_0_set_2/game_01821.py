
# Generated: 2025-08-28T02:48:34.412011
# Source Brief: brief_01821.md
# Brief Index: 1821

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to squash a bug."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade action. Squash swarms of colorful bugs with your cursor before they reach the bottom or time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    CURSOR_SPEED = 15

    COLOR_BG = (173, 216, 230) # Light Blue
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    
    BUG_COLORS = {
        'green': (50, 205, 50),   # Slow, standard points
        'red': (255, 69, 0),      # Fast, standard points
        'blue': (30, 144, 255),   # Special: Double points
        'yellow': (255, 215, 0),  # Special: Time bonus
    }
    BUG_OUTLINE_COLOR = (40, 40, 40)
    
    STAGE_CONFIG = {
        1: {'num_bugs': 30, 'speed_multiplier': 1.0, 'time_limit': 60.0, 'bug_types': ['green', 'red']},
        2: {'num_bugs': 50, 'speed_multiplier': 1.2, 'time_limit': 60.0, 'bug_types': ['green', 'red']},
        3: {'num_bugs': 70, 'speed_multiplier': 1.5, 'time_limit': 60.0, 'bug_types': ['green', 'red', 'blue', 'yellow']},
    }
    
    class Bug:
        def __init__(self, bug_type, speed_multiplier, np_random):
            self.pos = [
                np_random.uniform(20, GameEnv.SCREEN_WIDTH - 20),
                np_random.uniform(-200, -20)
            ]
            self.type = bug_type
            self.color = GameEnv.BUG_COLORS[bug_type]
            base_speed = np_random.uniform(0.8, 1.5) if bug_type != 'red' else np_random.uniform(1.8, 2.5)
            self.speed = base_speed * speed_multiplier
            self.size = np_random.integers(10, 15)
            self.hitbox_size = self.size * 1.2

        def move(self):
            self.pos[1] += self.speed

    class Particle:
        def __init__(self, pos, color, np_random):
            self.pos = list(pos)
            angle = np_random.uniform(0, 2 * math.pi)
            speed = np_random.uniform(1, 5)
            self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.lifetime = np_random.integers(10, 20)
            self.color = color
            self.size = np_random.uniform(2, 5)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)
        
        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_remaining = 0.0
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.bugs = []
        self.particles = []
        self.prev_space_held = False
        self.dt = 1.0 / self.FPS
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.particles = []
        self.prev_space_held = False

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        config = self.STAGE_CONFIG[self.stage]
        self.time_remaining = config['time_limit']
        self.bugs = []
        
        num_bugs = config['num_bugs']
        bug_types = config['bug_types']
        speed_multiplier = config['speed_multiplier']

        for _ in range(num_bugs):
            bug_type = self.np_random.choice(bug_types)
            self.bugs.append(self.Bug(bug_type, speed_multiplier, self.np_random))

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for existing
        self.game_over = False
        
        # 1. Handle player actions
        self._handle_input(movement, space_held)

        # 2. Check for click event and process squashes
        click_occurred = space_held and not self.prev_space_held
        if click_occurred:
            squashed_bugs, squash_reward = self._process_squash()
            reward += squash_reward
            if not squashed_bugs:
                reward -= 0.1 # Penalty for missing

        self.prev_space_held = space_held

        # 3. Update game state
        self._update_bugs()
        self._update_particles()
        self.time_remaining -= self.dt
        
        # 4. Check for termination and progression
        # Loss condition: Bug reached bottom
        for bug in self.bugs:
            if bug.pos[1] + bug.size > self.SCREEN_HEIGHT:
                self.game_over = True
                reward = -100
                break
        
        # Loss condition: Time ran out
        if self.time_remaining <= 0 and not self.game_over:
            self.time_remaining = 0
            self.game_over = True
            reward = -100

        # Win condition: All bugs squashed
        if not self.bugs and not self.game_over:
            if self.stage < len(self.STAGE_CONFIG):
                reward += 50  # Stage clear bonus
                self.stage += 1
                self._setup_stage()
            else:
                self.win = True
                self.game_over = True
                reward += 100 # Game win bonus

        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _process_squash(self):
        squashed_something = False
        reward = 0
        
        # Iterate backwards to allow safe removal
        for i in range(len(self.bugs) - 1, -1, -1):
            bug = self.bugs[i]
            dist = math.hypot(self.cursor_pos[0] - bug.pos[0], self.cursor_pos[1] - bug.pos[1])
            
            if dist < bug.hitbox_size:
                # Create particles
                for _ in range(15):
                    self.particles.append(self.Particle(bug.pos, bug.color, self.np_random))
                
                # Grant rewards/bonuses
                if bug.type == 'blue':
                    reward += 5
                elif bug.type == 'yellow':
                    reward += 2
                    self.time_remaining = min(self.STAGE_CONFIG[self.stage]['time_limit'], self.time_remaining + 3.0)
                else:
                    reward += 1
                
                self.score += int(reward)
                self.bugs.pop(i)
                squashed_something = True
                # sound_effect: "squish.wav"
                break # Only squash one bug per click
        
        return squashed_something, reward
        
    def _update_bugs(self):
        for bug in self.bugs:
            bug.move()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.pos[0] += p.vel[0]
            p.pos[1] += p.vel[1]
            p.lifetime -= 1
            p.size = max(0, p.size - 0.15)
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.pos[0]), int(p.pos[1])), int(p.size))

        # Render bugs
        for bug in self.bugs:
            pos = (int(bug.pos[0]), int(bug.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bug.size, bug.color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bug.size, self.BUG_OUTLINE_COLOR)

        # Render cursor
        if not self.game_over:
            cursor_pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
            radius = 15
            pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
            
            # Outer ring
            pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], radius, (255, 255, 255))
            pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], radius-1, (0,0,0))
            
            # Crosshairs
            pygame.draw.line(self.screen, (255, 255, 255), (cursor_pos[0] - 5, cursor_pos[1]), (cursor_pos[0] + 5, cursor_pos[1]), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (cursor_pos[0], cursor_pos[1] - 5), (cursor_pos[0], cursor_pos[1] + 5), 1)

    def _render_ui(self):
        # Helper to render text with shadow
        def draw_text(text, font, color, pos, shadow_color, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = pos
            else:
                text_rect.topleft = pos
            
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Score
        draw_text(f"Score: {self.score}", self.font_ui, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Time
        time_str = f"Time: {max(0, int(self.time_remaining))}"
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_TEXT
        draw_text(time_str, self.font_ui, time_color, (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT_SHADOW)

        # Stage
        draw_text(f"Stage: {self.stage}", self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, 10), self.COLOR_TEXT_SHADOW, center=True)

        # Game Over / Win Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            draw_text(message, self.font_msg, color, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_TEXT_SHADOW, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": self.time_remaining,
            "bugs_remaining": len(self.bugs)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # --- Manual Play ---
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Bug Squasher")
    # clock = pygame.time.Clock()
    
    # running = True
    # total_reward = 0
    # while running:
    #     movement = 0 # no-op
    #     space = 0
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
        
    #     action = [movement, space, 0]
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
        
    #     # Render to screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    #         pygame.time.wait(2000)
    #         total_reward = 0
    #         env.reset()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
        
    #     clock.tick(env.FPS)
    
    # env.close()

    # --- Validation Run ---
    env.validate_implementation()
    obs, info = env.reset()
    print("Initial Info:", info)
    terminated = False
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i % 50 == 0):
            print(f"Step {i}, Reward: {reward:.2f}, Info: {info}")
        if terminated:
            print(f"Episode terminated at step {i}.")
            break
    env.close()