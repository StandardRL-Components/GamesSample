import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:48:03.480434
# Source Brief: brief_00019.md
# Brief Index: 19
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim and release pendulums to hit all the targets before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ to select a pendulum and ←→ to aim. Press space to release the pendulums and shift to reset their positions."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 180, 0)
    COLOR_TIMER_CRIT = (255, 80, 80)
    COLOR_TARGET = (255, 215, 0)
    COLOR_TARGET_HIT = (80, 200, 120)
    COLOR_PENDULUM_1 = (255, 80, 80)
    COLOR_PENDULUM_2 = (80, 255, 80)
    COLOR_PENDULUM_3 = (80, 150, 255)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    GRAVITY = 9.8
    DAMPING = 0.998
    AIM_ADJUST_SPEED = 0.02 # Radians per step

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_info = pygame.font.SysFont("monospace", 14)

        self.game_phase = "aiming"
        self.pendulums = []
        self.targets = []
        self.particles = []
        self.selected_pendulum_idx = 0
        self.steps = 0
        self.score = 0
        self.time_remaining = 0.0
        self.min_distances = {}
        self.terminated = False

        # Note: reset() is not called in __init__ as per Gymnasium standard practice.
        # It's expected to be called by the user after environment creation.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.terminated = False
        self.game_phase = "aiming"
        self.time_remaining = float(self.GAME_DURATION_SECONDS)
        self.particles.clear()

        # Initialize Pendulums with different lengths to create different frequencies
        self.pendulums = [
            self._create_pendulum(self.SCREEN_WIDTH * 0.25, 120, self.COLOR_PENDULUM_1, 1.0),
            self._create_pendulum(self.SCREEN_WIDTH * 0.50, 140, self.COLOR_PENDULUM_2, 1.0),
            self._create_pendulum(self.SCREEN_WIDTH * 0.75, 100, self.COLOR_PENDULUM_3, 1.0),
        ]
        self.selected_pendulum_idx = 0

        # Initialize Targets
        self.targets = [
            {"pos": (100, 300), "radius": 15, "hit": False},
            {"pos": (220, 350), "radius": 20, "hit": False},
            {"pos": (320, 280), "radius": 18, "hit": False},
            {"pos": (450, 360), "radius": 22, "hit": False},
            {"pos": (550, 300), "radius": 15, "hit": False},
        ]
        
        # Initialize distance tracking for rewards
        self.min_distances = {i: float('inf') for i in range(len(self.targets))}
        for i, target in enumerate(self.targets):
            for pendulum in self.pendulums:
                bob_pos = self._get_bob_position(pendulum)
                dist = self._distance(bob_pos, target['pos'])
                if dist < self.min_distances[i]:
                    self.min_distances[i] = dist

        return self._get_observation(), self._get_info()

    def _create_pendulum(self, pivot_x, length, color, initial_angle_factor):
        return {
            "pivot": (pivot_x, 50),
            "length": length,
            "angle": math.pi * initial_angle_factor, # Start pointing down
            "angular_velocity": 0.0,
            "color": color,
            "bob_radius": 12,
        }

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small time penalty

        # --- Handle Actions ---
        if self.game_phase == "aiming":
            if space_held:
                self.game_phase = "swinging"
                # // Play "swing_start" sound
            elif shift_held:
                for p in self.pendulums:
                    p["angle"] = math.pi
                # // Play "reset_aim" sound
            else:
                # Change selected pendulum
                if movement == 1: # Up
                    self.selected_pendulum_idx = (self.selected_pendulum_idx - 1) % len(self.pendulums)
                elif movement == 2: # Down
                    self.selected_pendulum_idx = (self.selected_pendulum_idx + 1) % len(self.pendulums)
                
                # Adjust angle of selected pendulum
                if movement == 3: # Left
                    self.pendulums[self.selected_pendulum_idx]['angle'] -= self.AIM_ADJUST_SPEED
                elif movement == 4: # Right
                    self.pendulums[self.selected_pendulum_idx]['angle'] += self.AIM_ADJUST_SPEED
                
                # Clamp angle to prevent full circles
                self.pendulums[self.selected_pendulum_idx]['angle'] = np.clip(
                    self.pendulums[self.selected_pendulum_idx]['angle'], math.pi/2, 3*math.pi/2
                )

        # --- Update Game State ---
        if self.game_phase == "swinging":
            self.time_remaining -= 1.0 / self.FPS
            
            # Update pendulums physics
            for pendulum in self.pendulums:
                # Simple harmonic motion approximation for pendulum physics
                angular_acceleration = -(self.GRAVITY / pendulum['length']) * math.sin(pendulum['angle'] - math.pi)
                pendulum['angular_velocity'] += angular_acceleration * (1.0 / self.FPS)
                pendulum['angular_velocity'] *= self.DAMPING
                pendulum['angle'] += pendulum['angular_velocity'] * (1.0 / self.FPS)

            # Check for collisions and calculate proximity reward
            for pendulum in self.pendulums:
                bob_pos = self._get_bob_position(pendulum)
                for i, target in enumerate(self.targets):
                    dist = self._distance(bob_pos, target['pos'])
                    
                    # Proximity reward for getting closer
                    if not target['hit'] and dist < self.min_distances[i]:
                        reward += 0.1
                        self.min_distances[i] = dist

                    # Collision detection
                    if not target['hit'] and dist < (pendulum['bob_radius'] + target['radius']):
                        target['hit'] = True
                        self.score += 20
                        reward += 20.0
                        self._create_particles(target['pos'])
                        # // Play "target_hit" sound

        self._update_particles()

        # --- Check Termination ---
        self.steps += 1
        all_targets_hit = all(t['hit'] for t in self.targets)
        time_up = self.time_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        self.terminated = all_targets_hit or time_up or max_steps_reached
        
        if self.terminated:
            if all_targets_hit:
                reward += 50.0 # Victory bonus
                # // Play "win_game" sound
            elif time_up or max_steps_reached:
                reward -= 50.0 # Failure penalty
                # // Play "lose_game" sound

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # Truncated is always False
            self._get_info()
        )

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
            "time_remaining": self.time_remaining,
            "targets_hit": sum(1 for t in self.targets if t['hit']),
            "game_phase": self.game_phase,
        }

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw targets
        for target in self.targets:
            color = self.COLOR_TARGET_HIT if target['hit'] else self.COLOR_TARGET
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            # Draw a soft glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], target['radius'] + 3, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], target['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], target['radius'], color)

        # Draw pendulums
        for i, pendulum in enumerate(self.pendulums):
            pivot_pos = (int(pendulum['pivot'][0]), int(pendulum['pivot'][1]))
            bob_pos = self._get_bob_position(pendulum)
            bob_pos_int = (int(bob_pos[0]), int(bob_pos[1]))
            
            # Draw arm
            pygame.draw.aaline(self.screen, pendulum['color'], pivot_pos, bob_pos_int, 2)
            
            # Draw pivot
            pygame.gfxdraw.filled_circle(self.screen, pivot_pos[0], pivot_pos[1], 5, pendulum['color'])
            pygame.gfxdraw.aacircle(self.screen, pivot_pos[0], pivot_pos[1], 5, pendulum['color'])

            # Draw bob glow
            pygame.gfxdraw.filled_circle(self.screen, bob_pos_int[0], bob_pos_int[1], pendulum['bob_radius'] + 4, (*pendulum['color'], 80))
            # Draw bob
            pygame.gfxdraw.filled_circle(self.screen, bob_pos_int[0], bob_pos_int[1], pendulum['bob_radius'], pendulum['color'])
            pygame.gfxdraw.aacircle(self.screen, bob_pos_int[0], bob_pos_int[1], pendulum['bob_radius'], pendulum['color'])
            
            # Highlight selected pendulum in aiming phase
            if self.game_phase == "aiming" and i == self.selected_pendulum_idx:
                pygame.gfxdraw.aacircle(self.screen, pivot_pos[0], pivot_pos[1], 10, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, pivot_pos[0], pivot_pos[1], 11, (255, 255, 255))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_color = self.COLOR_UI_TEXT
        if self.time_remaining < 10: time_color = self.COLOR_TIMER_WARN
        if self.time_remaining < 5: time_color = self.COLOR_TIMER_CRIT
        timer_text = self.font_ui.render(f"TIME: {self.time_remaining:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Instructions
        if self.game_phase == "aiming":
            info_text = self.font_info.render("UP/DOWN: Select | LEFT/RIGHT: Aim | SPACE: Swing | SHIFT: Reset", True, self.COLOR_UI_TEXT)
            self.screen.blit(info_text, (self.SCREEN_WIDTH // 2 - info_text.get_width() // 2, self.SCREEN_HEIGHT - 20))

    def _get_bob_position(self, pendulum):
        x = pendulum['pivot'][0] + pendulum['length'] * math.sin(pendulum['angle'])
        y = pendulum['pivot'][1] + pendulum['length'] * math.cos(pendulum['angle'])
        return x, y

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _create_particles(self, position):
        # // Play "particle_burst" sound
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(1, 4),
                'lifespan': random.randint(20, 40),
                'color': (
                    random.randint(200, 255),
                    random.randint(200, 255),
                    random.randint(180, 255)
                )
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.05
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # --- Example Usage for human play ---
    # Un-comment the following line to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    # The validate_implementation method is for developer use, not part of the standard API
    # env.validate_implementation()
    
    obs, info = env.reset()
    terminated = False
    
    # Create a display for human play if not in headless mode
    pygame_screen = None
    if "SDL_VIDEODRIVER" not in os.environ:
        pygame_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Pendulum Precision")
    
    clock = pygame.time.Clock()
    total_reward = 0
    
    # Game loop for human control
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # This event loop is for human interaction, not required for the agent
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if pygame_screen:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        else: # Simple auto-play for headless mode testing
            action = env.action_space.sample()

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if pygame_screen:
            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            pygame_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            if pygame_screen:
                pygame.time.wait(2000) # Pause for 2 seconds before resetting
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()