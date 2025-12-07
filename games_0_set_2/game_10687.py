import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:55.047004
# Source Brief: brief_00687.md
# Brief Index: 687
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects like sparks."""
    def __init__(self, pos, np_random, color=(255, 60, 60), initial_life=30):
        self.x, self.y = pos
        self.np_random = np_random
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(2, 7)
        
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        self.life = self.np_random.uniform(initial_life * 0.7, initial_life * 1.3)
        self.initial_life = self.life
        self.color = color

    def update(self):
        """Update particle position and life."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        """Draw the particle on the screen."""
        if self.life > 0:
            # Fade out and shrink
            alpha = max(0, min(255, int(255 * (self.life / self.initial_life))))
            radius = max(0, int(5 * (self.life / self.initial_life)))
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
            surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)))

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player manipulates swinging pendulums to prevent
    collisions and maintain rhythmic harmony.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Keep a set of swinging pendulums from colliding. Apply impulses to change their rhythm and survive as long as possible."
    )
    user_guide = (
        "Use ↑/↓ arrow keys to select a pendulum. Use ←/→ to push the selected pendulum. Press space to reset a pendulum to its resting position."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # === Game Constants ===
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.DT = 1.0 / self.FPS
        self.NUM_PENDULUMS = 4
        self.GRAVITY = 9.8
        self.DAMPING = 0.999
        self.IMPULSE_STRENGTH = 0.05
        self.WIN_TIME = 60.0
        self.MAX_STEPS = int(self.WIN_TIME * self.FPS * 1.5) # ~90 seconds

        # === Visuals ===
        self.COLOR_BG = (26, 26, 46)
        self.PENDULUM_COLORS = [
            (0, 168, 255), (0, 200, 200), (100, 150, 255), (50, 120, 220)
        ]
        self.COLOR_SPARK = (255, 80, 80)
        self.COLOR_SELECT_GLOW = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # === Gymnasium Spaces ===
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # === Pygame Setup ===
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("dejavusansmono", 22, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont("monospace", 22, bold=True)
        
        # === State Variables (initialized in reset) ===
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_survived = None
        self.pendulums = None
        self.selected_pendulum_idx = None
        self.particles = None
        self.last_time_bucket = None
        
        self.reset()
        # self.validate_implementation() # This can be removed or commented out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_survived = 0.0
        self.last_time_bucket = 0
        self.selected_pendulum_idx = 0
        self.particles = []
        
        self._initialize_pendulums()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        reward = 0.0
        
        self._handle_actions(action)
        self._update_physics()
        self._update_particles()
        
        collision_detected = self._check_collisions()
        
        if not collision_detected:
            self.time_survived += self.DT
            reward += 0.1  # Survival reward

            current_time_bucket = int(self.time_survived // 2)
            if current_time_bucket > self.last_time_bucket:
                reward += 1.0 * (current_time_bucket - self.last_time_bucket)
                self.last_time_bucket = current_time_bucket
        
        win = self.time_survived >= self.WIN_TIME
        if collision_detected or win:
            self.game_over = True
            if win:
                reward = 100.0  # Win reward
            else:
                reward = -100.0  # Collision penalty
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _initialize_pendulums(self):
        self.pendulums = []
        pivots_x = [self.WIDTH // (self.NUM_PENDULUMS + 1) * (i + 1) for i in range(self.NUM_PENDULUMS)]
        lengths = [150, 200, 180, 220]
        bob_radii = [15, 20, 18, 22]
        
        for i in range(self.NUM_PENDULUMS):
            self.pendulums.append({
                "pivot": (pivots_x[i], 80),
                "length": lengths[i] + self.np_random.uniform(-10, 10),
                "angle": self.np_random.uniform(-math.pi / 6, math.pi / 6),
                "angular_velocity": 0.0,
                "bob_radius": bob_radii[i],
                "color": self.PENDULUM_COLORS[i]
            })

    def _handle_actions(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        if movement in [1, 2]: # Up/Down selection
            # Debounce selection change to one per press
            if not hasattr(self, '_last_movement') or self._last_movement == 0:
                if movement == 1: # Up
                    self.selected_pendulum_idx = (self.selected_pendulum_idx - 1 + self.NUM_PENDULUMS) % self.NUM_PENDULUMS
                elif movement == 2: # Down
                    self.selected_pendulum_idx = (self.selected_pendulum_idx + 1) % self.NUM_PENDULUMS
        self._last_movement = movement

        selected_p = self.pendulums[self.selected_pendulum_idx]

        if movement == 3: # Left
            selected_p["angular_velocity"] -= self.IMPULSE_STRENGTH
        elif movement == 4: # Right
            selected_p["angular_velocity"] += self.IMPULSE_STRENGTH
            
        if space_pressed:
            selected_p["angle"] = 0.0
            selected_p["angular_velocity"] = 0.0
            # // Sound: Whoosh reset

    def _update_physics(self):
        for p in self.pendulums:
            angular_acceleration = -(self.GRAVITY / p["length"]) * math.sin(p["angle"])
            p["angular_velocity"] += angular_acceleration * self.DT
            p["angular_velocity"] *= self.DAMPING
            p["angle"] += p["angular_velocity"] * self.DT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_collisions(self):
        bob_positions = []
        for p in self.pendulums:
            x = p["pivot"][0] + p["length"] * math.sin(p["angle"])
            y = p["pivot"][1] + p["length"] * math.cos(p["angle"])
            bob_positions.append((x, y, p["bob_radius"]))

        for i in range(self.NUM_PENDULUMS):
            for j in range(i + 1, self.NUM_PENDULUMS):
                pos1, r1 = bob_positions[i][:2], bob_positions[i][2]
                pos2, r2 = bob_positions[j][:2], bob_positions[j][2]
                dist_sq = (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
                
                if dist_sq < (r1 + r2)**2:
                    collision_point = ((pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2)
                    self._create_sparks(collision_point)
                    # // Sound: CRACK!
                    return True
        return False
        
    def _create_sparks(self, position):
        num_sparks = 30
        for _ in range(num_sparks):
            self.particles.append(Particle(position, self.np_random, color=self.COLOR_SPARK))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)

        selected_p = self.pendulums[self.selected_pendulum_idx]
        pivot = selected_p["pivot"]
        for i in range(20, 0, -2):
            alpha = max(0, 120 - i * (120/20))
            color = (*self.COLOR_SELECT_GLOW, alpha)
            s = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (i, i), i)
            self.screen.blit(s, (int(pivot[0] - i), int(pivot[1] - i)))

        for p in self.pendulums:
            pivot_pos = (int(p["pivot"][0]), int(p["pivot"][1]))
            bob_radius = int(p["bob_radius"])
            
            bob_x = pivot_pos[0] + p["length"] * math.sin(p["angle"])
            bob_y = pivot_pos[1] + p["length"] * math.cos(p["angle"])
            bob_pos = (int(bob_x), int(bob_y))
            
            pygame.draw.aaline(self.screen, p["color"], pivot_pos, bob_pos, 2)
            
            pygame.gfxdraw.filled_circle(self.screen, bob_pos[0], bob_pos[1], bob_radius, p["color"])
            pygame.gfxdraw.aacircle(self.screen, bob_pos[0], bob_pos[1], bob_radius, p["color"])
            
            pygame.gfxdraw.filled_circle(self.screen, pivot_pos[0], pivot_pos[1], 5, self.COLOR_UI_TEXT)
            pygame.gfxdraw.aacircle(self.screen, pivot_pos[0], pivot_pos[1], 5, self.COLOR_UI_TEXT)

    def _render_ui(self):
        time_text = f"TIME: {self.time_survived:.2f} / {self.WIN_TIME:.0f}s"
        time_surf = self.font.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_survived": self.time_survived}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen to be a display window
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pendulum Harmony")

    terminated = False
    total_reward = 0.0
    
    # Game loop
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Manually render to the display
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # So we transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Time Survived: {info['time_survived']:.2f}s")
            # Wait a moment before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()