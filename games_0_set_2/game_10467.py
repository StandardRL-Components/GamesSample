import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:31:46.773599
# Source Brief: brief_00467.md
# Brief Index: 467
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a squadron of 4 drones
    to protect a central core from incoming projectiles. The drones form a
    dynamic shield, and their positioning is key to survival.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a squadron of four drones to form a dynamic shield and protect your central core from waves of incoming projectiles."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the currently selected drone. The selected drone cycles automatically after each move."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3100  # 5 waves * 20 seconds/wave * 30 FPS + buffer

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_CORE = (100, 150, 255)
    COLOR_CORE_GLOW = (50, 80, 150)
    COLOR_DRONE = (0, 255, 150)
    COLOR_DRONE_SELECTED = (255, 255, 255)
    COLOR_PROJECTILE = (255, 80, 80)
    COLOR_SHIELD = (0, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 120, 120)

    # Game Parameters
    CORE_RADIUS = 20
    DRONE_SIZE = 12
    DRONE_SPEED = 8.0
    PROJECTILE_RADIUS = 5
    SHIELD_MAX_DIST = 150
    NUM_DRONES = 4
    WAVE_DURATION = 600  # 20 seconds at 30 FPS
    TOTAL_WAVES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.core_pos = None
        self.drones = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.selected_drone_idx = 0
        self.current_wave = 1
        self.steps_in_wave = 0
        self.score_multiplier = 1.0
        self.last_shield_strength = 0

        # This is a critical self-check
        # self.validate_implementation() # Commented out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.selected_drone_idx = 0
        self.current_wave = 1
        self.steps_in_wave = 0
        self.score_multiplier = 1.0
        self.last_shield_strength = 0

        # Initialize Game Objects
        self.core_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        
        self.drones = [
            {'pos': pygame.Vector2(self.core_pos.x - 50, self.core_pos.y - 50)},
            {'pos': pygame.Vector2(self.core_pos.x + 50, self.core_pos.y - 50)},
            {'pos': pygame.Vector2(self.core_pos.x + 50, self.core_pos.y + 50)},
            {'pos': pygame.Vector2(self.core_pos.x - 50, self.core_pos.y + 50)},
        ]
        
        self.projectiles.clear()
        self.particles.clear()
        
        # Create a static starfield for visual appeal
        if not self.stars:
             self.stars = [
                (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
                for _ in range(100)
            ]

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.steps_in_wave += 1
        reward = 0.01  # Survival reward

        # --- 1. Handle Player Input ---
        movement = action[0]
        
        if movement > 0: # Any move action cycles the selected drone
            move_vec = {
                1: pygame.Vector2(0, -1), # Up
                2: pygame.Vector2(0, 1),  # Down
                3: pygame.Vector2(-1, 0), # Left
                4: pygame.Vector2(1, 0),  # Right
            }[movement]
            
            drone = self.drones[self.selected_drone_idx]
            drone['pos'] += move_vec * self.DRONE_SPEED
            
            # Clamp drone position to screen bounds
            drone['pos'].x = max(self.DRONE_SIZE / 2, min(self.WIDTH - self.DRONE_SIZE / 2, drone['pos'].x))
            drone['pos'].y = max(self.DRONE_SIZE / 2, min(self.HEIGHT - self.DRONE_SIZE / 2, drone['pos'].y))
            
            self.selected_drone_idx = (self.selected_drone_idx + 1) % self.NUM_DRONES
            # Sound placeholder: # sfx_drone_move()

        # --- 2. Update Game Logic ---
        self._update_projectiles()
        self._update_particles()
        
        # --- 3. Collision Detection ---
        reward += self._handle_collisions()
        
        # --- 4. Wave Progression ---
        if not self.game_over and self.steps_in_wave >= self.WAVE_DURATION:
            self.current_wave += 1
            if self.current_wave > self.TOTAL_WAVES:
                self.win = True
                self.game_over = True
                reward += 100.0 # Victory reward
                # Sound placeholder: # sfx_win_game()
            else:
                reward += 1.0 # Wave complete reward
                self.score_multiplier *= 1.5
                self.steps_in_wave = 0
                self._spawn_wave()
                # Sound placeholder: # sfx_wave_complete()

        # --- 5. Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        # Final reward on loss
        if self.game_over and not self.win:
            reward = -100.0
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_wave(self):
        num_projectiles = 3 + (self.current_wave - 1) * 2
        speed = 1.5 + (self.current_wave - 1) * 0.2
        
        for _ in range(num_projectiles):
            # Spawn projectile from a random edge
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = pygame.Vector2(random.randint(0, self.WIDTH), -self.PROJECTILE_RADIUS)
            elif edge == 'bottom':
                pos = pygame.Vector2(random.randint(0, self.WIDTH), self.HEIGHT + self.PROJECTILE_RADIUS)
            elif edge == 'left':
                pos = pygame.Vector2(-self.PROJECTILE_RADIUS, random.randint(0, self.HEIGHT))
            else: # right
                pos = pygame.Vector2(self.WIDTH + self.PROJECTILE_RADIUS, random.randint(0, self.HEIGHT))
            
            # Aim towards the core with some randomness
            target_pos = self.core_pos + pygame.Vector2(random.uniform(-50, 50), random.uniform(-50, 50))
            vel = (target_pos - pos).normalize() * speed
            
            self.projectiles.append({'pos': pos, 'vel': vel})
        # Sound placeholder: # sfx_wave_start()

    def _update_projectiles(self):
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            # Despawn if off-screen
            if -20 < p['pos'].x < self.WIDTH + 20 and -20 < p['pos'].y < self.HEIGHT + 20:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = set()

        for i, p in enumerate(self.projectiles):
            # Projectile-Core Collision
            if p['pos'].distance_to(self.core_pos) < self.CORE_RADIUS + self.PROJECTILE_RADIUS:
                self.game_over = True
                self._create_explosion(self.core_pos, self.COLOR_CORE, 50)
                # Sound placeholder: # sfx_core_destroyed()
                break # Game over, no need to check more

            # Projectile-Drone Collision
            for drone in self.drones:
                if p['pos'].distance_to(drone['pos']) < self.DRONE_SIZE / 2 + self.PROJECTILE_RADIUS:
                    if i not in projectiles_to_remove:
                        projectiles_to_remove.add(i)
                        reward += 0.1 * self.score_multiplier
                        self.score += 10 * self.score_multiplier
                        self._create_explosion(p['pos'], self.COLOR_PARTICLE, 20)
                        # Sound placeholder: # sfx_projectile_destroyed()

            if i in projectiles_to_remove:
                continue

            # Projectile-Shield Collision
            for j in range(self.NUM_DRONES):
                for k in range(j + 1, self.NUM_DRONES):
                    d1_pos = self.drones[j]['pos']
                    d2_pos = self.drones[k]['pos']
                    if d1_pos.distance_to(d2_pos) < self.SHIELD_MAX_DIST:
                        if self._line_circle_collision(d1_pos, d2_pos, p['pos'], self.PROJECTILE_RADIUS):
                            if i not in projectiles_to_remove:
                                projectiles_to_remove.add(i)
                                reward += 0.1 * self.score_multiplier
                                self.score += 10 * self.score_multiplier
                                self._create_explosion(p['pos'], self.COLOR_PARTICLE, 20)
                                # Sound placeholder: # sfx_shield_hit()
                                break
                if i in projectiles_to_remove:
                    break

        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        
        return reward

    def _line_circle_collision(self, p1, p2, circle_pos, circle_radius):
        line_vec = p2 - p1
        if line_vec.length() == 0: return False
        
        t = ((circle_pos - p1).dot(line_vec)) / line_vec.dot(line_vec)
        t = max(0, min(1, t)) # Clamp to segment
        
        closest_point = p1 + t * line_vec
        return closest_point.distance_to(circle_pos) < circle_radius

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (200, 200, 200), (x, y, size, size))

        # Render Core
        core_glow_radius = int(self.CORE_RADIUS * (1.5 + 0.1 * math.sin(self.steps * 0.1)))
        pygame.gfxdraw.filled_circle(self.screen, int(self.core_pos.x), int(self.core_pos.y), core_glow_radius, self.COLOR_CORE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.core_pos.x), int(self.core_pos.y), self.CORE_RADIUS, self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.screen, int(self.core_pos.x), int(self.core_pos.y), self.CORE_RADIUS, self.COLOR_CORE)

        # Render Shield
        shield_strength = 0
        active_links = 0
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                p1 = self.drones[i]['pos']
                p2 = self.drones[j]['pos']
                dist = p1.distance_to(p2)
                if dist < self.SHIELD_MAX_DIST:
                    alpha = int(200 * (1 - dist / self.SHIELD_MAX_DIST))
                    pygame.draw.aaline(self.screen, (*self.COLOR_SHIELD, alpha), p1, p2, 2)
                    shield_strength += (self.SHIELD_MAX_DIST - dist)
                    active_links += 1
        
        self.last_shield_strength = (shield_strength / (self.SHIELD_MAX_DIST * 6)) * 100 if active_links > 0 else 0


        # Render Projectiles
        for p in self.projectiles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Render Drones
        for i, drone in enumerate(self.drones):
            pos_int = (int(drone['pos'].x), int(drone['pos'].y))
            size = self.DRONE_SIZE
            rect = pygame.Rect(pos_int[0] - size / 2, pos_int[1] - size / 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_DRONE, rect, border_radius=2)
            if i == self.selected_drone_idx:
                 pygame.draw.rect(self.screen, self.COLOR_DRONE_SELECTED, rect, width=2, border_radius=2)
        
        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(self.PROJECTILE_RADIUS * (p['lifespan'] / p['max_life']))
            if size > 0:
                pos_int = (int(p['pos'].x), int(p['pos'].y))
                try:
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], max(0, size), color)
                except OverflowError: # Can happen if alpha is invalid
                    pass

        # Render Game Over/Win Text
        if self.game_over:
            text = "VICTORY" if self.win else "CORE DESTROYED"
            color = self.COLOR_DRONE if self.win else self.COLOR_PROJECTILE
            surf = self.font_big.render(text, True, color)
            self.screen.blit(surf, surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Shield Strength
        strength_text = self.font_ui.render(f"SHIELD: {int(self.last_shield_strength)}%", True, self.COLOR_TEXT)
        self.screen.blit(strength_text, strength_text.get_rect(centerx=self.WIDTH/2, y=self.HEIGHT - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "win": self.win,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- To run and play the game manually ---
if __name__ == "__main__":
    # Unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate display for human playing
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Drone Shield")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Key mapping
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, action_val in key_to_action.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
        if keys[pygame.K_r]: # Reset
            obs, info = env.reset()
            done = False
        if keys[pygame.K_ESCAPE]:
            running = False

        # --- Environment Step ---
        if not done:
            action = [movement_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()