
# Generated: 2025-08-27T22:20:43.021553
# Source Brief: brief_03092.md
# Brief Index: 3092

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for explosion particles
class Particle:
    def __init__(self, pos, color, speed, angle, lifetime):
        self.x, self.y = pos
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.vx = math.cos(angle) * speed * (0.8 + random.random() * 0.4)
        self.vy = math.sin(angle) * speed * (0.8 + random.random() * 0.4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95  # friction
        self.vy *= 0.95
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            # Fade out effect
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            alpha = max(0, min(255, alpha))
            temp_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*self.color, alpha), (2, 2), 2)
            surface.blit(temp_surface, (int(self.x) - 2, int(self.y) - 2))

# Helper class for shot tracer lines
class Tracer:
    def __init__(self, start_pos, end_pos, color, lifetime):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime)**2)
            alpha = max(0, min(255, alpha))
            pygame.draw.aaline(surface, (*self.color, alpha), self.start_pos, self.end_pos, 2)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the crosshair. Press space to fire."
    )

    game_description = (
        "A fast-paced, top-down target practice game. "
        "Shoot all the targets before time runs out to win. "
        "Accuracy and speed are key to a high score."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    TOTAL_TARGETS = 25
    TARGET_RADIUS = 20
    TARGET_SPAWN_BORDER = 30

    CROSSHAIR_SPEED = 12
    CROSSHAIR_SIZE = 15

    # Rewards
    REWARD_HIT = 0.5
    REWARD_MISS = -0.1
    REWARD_WIN = 10.0
    REWARD_LOSE = -10.0

    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_TARGET = (255, 80, 80)
    COLOR_TARGET_OUTLINE = (200, 50, 50)
    COLOR_CROSSHAIR = (220, 220, 255)
    COLOR_TRACER = (255, 255, 255)
    COLOR_PARTICLE = (255, 220, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 150, 0)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 50, bold=True)

        self.particles = []
        self.tracers = []

        self.reset()
        
        # This can be commented out for performance in production
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.time_remaining = self.MAX_STEPS
        self.targets_remaining = self.TOTAL_TARGETS

        self.crosshair_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self._spawn_target()

        self.particles.clear()
        self.tracers.clear()
        
        self.last_space_held = False
        self.last_shot_outcome = "none" # "hit", "miss", "none"

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1

            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_movement(movement)
            
            shot_fired = space_held and not self.last_space_held
            if shot_fired:
                reward += self._fire_shot()
            self.last_space_held = space_held

            self._update_dynamics()
            
            terminated = self.time_remaining <= 0 or self.targets_remaining <= 0 or self.steps >= self.MAX_STEPS

            if terminated:
                self.game_over = True
                if self.targets_remaining <= 0:
                    reward += self.REWARD_WIN
                    self.score += self.REWARD_WIN
                else: # Time ran out
                    reward += self.REWARD_LOSE
                    self.score += self.REWARD_LOSE

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.crosshair_pos[1] -= self.CROSSHAIR_SPEED
        elif movement == 2: # Down
            self.crosshair_pos[1] += self.CROSSHAIR_SPEED
        elif movement == 3: # Left
            self.crosshair_pos[0] -= self.CROSSHAIR_SPEED
        elif movement == 4: # Right
            self.crosshair_pos[0] += self.CROSSHAIR_SPEED
        
        # Clamp crosshair position to screen bounds
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.SCREEN_WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.SCREEN_HEIGHT)

    def _fire_shot(self):
        # sfx: Laser shot
        self.tracers.append(Tracer(self.crosshair_pos.copy(), self.target_pos.copy(), self.COLOR_TRACER, 10))
        
        distance_to_target = np.linalg.norm(self.crosshair_pos - self.target_pos)
        
        if distance_to_target <= self.TARGET_RADIUS:
            # sfx: Target hit explosion
            self.score += self.REWARD_HIT
            self.targets_remaining -= 1
            self._create_explosion(self.target_pos)
            if self.targets_remaining > 0:
                self._spawn_target()
            self.last_shot_outcome = "hit"
            return self.REWARD_HIT
        else:
            # sfx: Miss sound
            self.score += self.REWARD_MISS
            self.last_shot_outcome = "miss"
            return self.REWARD_MISS

    def _spawn_target(self):
        border = self.TARGET_SPAWN_BORDER + self.TARGET_RADIUS
        self.target_pos[0] = self.np_random.integers(border, self.SCREEN_WIDTH - border)
        self.target_pos[1] = self.np_random.integers(border, self.SCREEN_HEIGHT - border)

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            lifetime = random.randint(15, 30)
            self.particles.append(Particle(pos, self.COLOR_PARTICLE, speed, angle, lifetime))

    def _update_dynamics(self):
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)
        
        for t in self.tracers[:]:
            t.update()
            if t.lifetime <= 0:
                self.tracers.remove(t)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw target if game is not won
        if self.targets_remaining > 0:
            tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
            pygame.gfxdraw.filled_circle(self.screen, tx, ty, self.TARGET_RADIUS, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, tx, ty, self.TARGET_RADIUS, self.COLOR_TARGET_OUTLINE)
        
        # Draw tracers and particles
        for t in self.tracers:
            t.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw crosshair
        ch_x, ch_y = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = self.CROSSHAIR_SIZE
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (ch_x - size, ch_y), (ch_x + size, ch_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (ch_x, ch_y - size), (ch_x, ch_y + size), 2)
        pygame.gfxdraw.aacircle(self.screen, ch_x, ch_y, size // 2, self.COLOR_CROSSHAIR)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Targets Remaining
        targets_text = self.font_ui.render(f"TARGETS: {self.targets_remaining}/{self.TOTAL_TARGETS}", True, self.COLOR_TEXT)
        self.screen.blit(targets_text, (10, 35))

        # Timer
        seconds = self.time_remaining / self.FPS
        timer_color = self.COLOR_TEXT if seconds > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_ui.render(f"TIME: {seconds:.2f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            if self.targets_remaining <= 0:
                msg = "MISSION COMPLETE"
                color = self.COLOR_WIN
            else:
                msg = "TIME'S UP"
                color = self.COLOR_LOSE
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_remaining": self.targets_remaining,
            "time_remaining_seconds": self.time_remaining / self.FPS,
            "last_shot_outcome": self.last_shot_outcome,
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
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
            
    env.close()
    
    # Example with rendering
    print("\n--- Starting interactive example ---")
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use your display driver
    
    interactive_env = GameEnv(render_mode="rgb_array")
    obs, info = interactive_env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Target Practice")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = interactive_env.reset()

        obs, reward, terminated, truncated, info = interactive_env.step(action)
        
        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = interactive_env.reset()

        clock.tick(GameEnv.FPS)

    interactive_env.close()