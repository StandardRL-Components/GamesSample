
# Generated: 2025-08-28T05:11:41.607416
# Source Brief: brief_05494.md
# Brief Index: 5494

        
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
        "Controls: ←→ to move the catcher. Catch the bugs before they hit the ground!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must catch falling bugs. "
        "The bugs fall faster as you catch more. Don't miss too many!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_HEIGHT = 50
    FPS = 30

    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_GROUND = (34, 139, 34)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (50, 50, 50)
    COLOR_MISS = (255, 0, 0)
    BUG_COLORS = [(255, 105, 180), (255, 165, 0), (148, 0, 211), (0, 255, 255)]

    # Game parameters
    WIN_SCORE = 30
    LOSE_MISSES = 5
    MAX_STEPS = 1000 * 3 # Adjusted for 30fps to be reasonable

    CATCHER_WIDTH = 80
    CATCHER_HEIGHT = 20
    CATCHER_SPEED = 10
    
    BUG_SPAWN_RATE = 45 # Lower is faster
    INITIAL_BUG_SPEED = 2.0
    BUG_SPEED_INCREASE_INTERVAL = 10
    BUG_SPEED_INCREMENT = 0.5 # Increased from 0.05 for more noticeable difficulty

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Game state variables
        self.catcher_rect = None
        self.bugs = []
        self.effects = []
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.bug_spawn_timer = 0
        self.current_bug_speed = self.INITIAL_BUG_SPEED

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.catcher_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.CATCHER_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.GROUND_HEIGHT - self.CATCHER_HEIGHT,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT,
        )
        
        self.bugs = []
        self.effects = []
        self.bug_spawn_timer = 0
        self.current_bug_speed = self.INITIAL_BUG_SPEED

        self._spawn_bug()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        # --- Reward Calculation (Part 1: Proximity) ---
        reward = 0
        nearest_bug = self._get_nearest_bug()
        dist_before = float('inf')
        if nearest_bug:
            dist_before = abs(self.catcher_rect.centerx - nearest_bug["pos"][0])

        # --- Update Game Logic ---
        # Move Catcher
        if movement == 3:  # Left
            self.catcher_rect.x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_rect.x += self.CATCHER_SPEED
        
        # Clamp catcher position
        self.catcher_rect.x = max(0, min(self.catcher_rect.x, self.SCREEN_WIDTH - self.CATCHER_WIDTH))

        # --- Reward Calculation (Part 2: Proximity) ---
        if nearest_bug:
            dist_after = abs(self.catcher_rect.centerx - nearest_bug["pos"][0])
            if dist_after < dist_before:
                reward += 0.01 # Small reward for moving closer
            elif dist_after > dist_before:
                reward -= 0.01 # Small penalty for moving away
        
        # Update Bugs
        for bug in self.bugs[:]:
            bug["pos"][1] += bug["speed"]
            
            # Check for Catch
            if self.catcher_rect.colliderect(pygame.Rect(bug["pos"][0] - bug["size"], bug["pos"][1] - bug["size"], bug["size"]*2, bug["size"]*2)):
                self.score += 1
                reward += 1.0
                self._create_catch_effect(bug["pos"], bug["color"])
                self.bugs.remove(bug)
                # Play catch sound
                
                # Increase difficulty
                if self.score > 0 and self.score % self.BUG_SPEED_INCREASE_INTERVAL == 0:
                    self.current_bug_speed += self.BUG_SPEED_INCREMENT

            # Check for Miss
            elif bug["pos"][1] > self.SCREEN_HEIGHT - self.GROUND_HEIGHT:
                self.misses += 1
                reward -= 1.0
                self._create_miss_effect(bug["pos"])
                self.bugs.remove(bug)
                # Play miss sound
        
        # Update Effects
        for effect in self.effects[:]:
            effect["life"] -= 1
            if effect["type"] == "particle":
                effect["pos"][0] += effect["vel"][0]
                effect["pos"][1] += effect["vel"][1]
                effect["vel"][1] += 0.1 # Gravity
            if effect["life"] <= 0:
                self.effects.remove(effect)
        
        # Spawn new bugs
        self.bug_spawn_timer += 1
        if self.bug_spawn_timer >= self.BUG_SPAWN_RATE:
            self.bug_spawn_timer = 0
            self._spawn_bug()

        self.steps += 1
        
        # Check Termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 10.0
            terminated = True
            self.game_over = True
        elif self.misses >= self.LOSE_MISSES:
            reward -= 10.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_nearest_bug(self):
        if not self.bugs:
            return None
        # Find bug closest to the ground
        return min(self.bugs, key=lambda b: self.SCREEN_HEIGHT - self.GROUND_HEIGHT - b["pos"][1])

    def _spawn_bug(self):
        bug = {
            "pos": [self.np_random.integers(20, self.SCREEN_WIDTH - 20), -10],
            "speed": self.current_bug_speed + self.np_random.uniform(-0.2, 0.2),
            "size": self.np_random.integers(6, 10),
            "color": random.choice(self.BUG_COLORS),
            "leg_angle": self.np_random.uniform(0, math.pi)
        }
        self.bugs.append(bug)

    def _create_catch_effect(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                "type": "particle",
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.integers(2, 4)
            }
            self.effects.append(particle)

    def _create_miss_effect(self, pos):
        effect = {
            "type": "miss",
            "pos": [pos[0], self.SCREEN_HEIGHT - self.GROUND_HEIGHT - 15],
            "life": 45,
            "color": self.COLOR_MISS
        }
        self.effects.append(effect)
    
    def _get_observation(self):
        # --- Rendering ---
        # Background
        self.screen.fill(self.COLOR_SKY)
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT))
        
        # Effects (render before main objects)
        for effect in self.effects:
            if effect["type"] == "particle":
                alpha = int(255 * (effect["life"] / 30))
                if alpha > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(effect["pos"][0]), int(effect["pos"][1]), effect["size"], (*effect["color"], alpha))
            elif effect["type"] == "miss":
                alpha = int(255 * (effect["life"] / 45))
                if alpha > 0:
                    size = 15
                    pos = effect["pos"]
                    line1_start = (pos[0] - size, pos[1] - size)
                    line1_end = (pos[0] + size, pos[1] + size)
                    line2_start = (pos[0] - size, pos[1] + size)
                    line2_end = (pos[0] + size, pos[1] - size)
                    
                    # Create a temporary surface for alpha blending
                    s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.line(s, (*effect["color"], alpha), (0, 0), (size*2, size*2), 4)
                    pygame.draw.line(s, (*effect["color"], alpha), (0, size*2), (size*2, 0), 4)
                    self.screen.blit(s, (pos[0] - size, pos[1] - size))

        # Bugs
        for bug in self.bugs:
            self._render_bug(bug)

        # Catcher
        self._render_catcher()

        # UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_bug(self, bug):
        pos_x, pos_y = int(bug["pos"][0]), int(bug["pos"][1])
        size = bug["size"]
        color = bug["color"]
        
        # Body
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, size, color)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, size, color)
        
        # Eyes
        eye_offset = size // 3
        eye_size = max(1, size // 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (pos_x - eye_offset, pos_y - eye_offset), eye_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (pos_x + eye_offset, pos_y - eye_offset), eye_size)

        # Legs (animated by sine wave of y position)
        leg_angle = math.sin(pos_y * 0.1) * 0.5 + bug["leg_angle"]
        for i in range(-1, 2, 2):
            pygame.draw.line(self.screen, color, (pos_x, pos_y), (pos_x + i * size * math.cos(leg_angle), pos_y + size * math.sin(leg_angle)), 2)
            pygame.draw.line(self.screen, color, (pos_x, pos_y), (pos_x + i * size * math.cos(leg_angle + 0.5), pos_y + size * math.sin(leg_angle + 0.5)), 2)
    
    def _render_catcher(self):
        # Handle
        handle_color = (139, 69, 19)
        pygame.draw.rect(self.screen, handle_color, self.catcher_rect)
        
        # Net (arc)
        net_color = (255, 255, 224)
        net_rect = pygame.Rect(self.catcher_rect.x, self.catcher_rect.y - self.CATCHER_HEIGHT // 2, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        pygame.draw.arc(self.screen, net_color, net_rect, math.pi, 2 * math.pi, 3)

    def _render_text(self, text, font, pos, color, shadow_color):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_small, (10, 10), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)
        self._render_text(f"Misses: {self.misses}/{self.LOSE_MISSES}", self.font_small, (10, 40), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)
        self._render_text(f"Goal: {self.WIN_SCORE}", self.font_small, (self.SCREEN_WIDTH - 120, 10), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 0) if self.score >= self.WIN_SCORE else (255, 0, 0)
            self._render_text(msg, self.font_large, (self.SCREEN_WIDTH // 2 - 120, self.SCREEN_HEIGHT // 2 - 30), color, self.COLOR_UI_SHADOW)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "bugs_on_screen": len(self.bugs),
            "current_bug_speed": self.current_bug_speed,
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Set up Pygame window for human play
    pygame.display.set_caption("Bug Catcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting for the human player
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()