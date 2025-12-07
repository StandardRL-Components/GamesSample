import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math
import random

# Helper class for visual effects
class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        if alpha > 0:
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    auto_advance = True
    game_description = (
        "Control a time-traveling agent who can flip gravity. Shoot temporal anomalies to restore the seasons while managing your energy."
    )
    user_guide = (
        "Controls: ←→ to move, ↑↓ to flip gravity. Press space to shoot and shift to activate your shield."
    )

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.NUM_LEVELS = 4

        # Action and Observation Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.human_screen = None
        if self.render_mode == "human":
            # This will fail if SDL_VIDEODRIVER is "dummy", but is needed for human mode
            if os.getenv("SDL_VIDEODRIVER") != "dummy":
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        
        # Fonts and Colors
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PROJECTILE = (0, 255, 255)
        self.COLOR_ANOMALY = (200, 0, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.SEASON_COLORS = [
            {"bg": (15, 20, 40), "platform": (180, 190, 210), "detail": (100, 110, 130)},  # Winter
            {"bg": (20, 40, 25), "platform": (110, 150, 90), "detail": (70, 100, 60)},    # Spring
            {"bg": (50, 30, 10), "platform": (210, 160, 80), "detail": (150, 110, 50)},    # Summer
            {"bg": (40, 20, 15), "platform": (160, 90, 70), "detail": (110, 60, 50)},     # Autumn
        ]

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.gravity_dir = None
        self.temporal_energy = None
        self.platforms = None
        self.anomalies = None
        self.projectiles = None
        self.particles = None
        self.energy_pickups = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_level_index = None
        self.skills_unlocked = None
        self.shield_active = None
        self.shield_timer = None
        self.shoot_cooldown = None
        self.flip_cooldown = None
        self.last_movement_dir = None
        
        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level_index = 0
        self.skills_unlocked = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_dir = 1  # 1 for down, -1 for up
        self.last_movement_dir = 1 # 1 for right, -1 for left
        
        self.temporal_energy = 100.0
        self.shield_active = False
        self.shield_timer = 0
        
        self.projectiles = []
        self.particles = []
        
        self.shoot_cooldown = 0
        self.flip_cooldown = 0

        self._generate_level(self.current_level_index)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0

        # --- Handle Input and Cooldowns ---
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.flip_cooldown = max(0, self.flip_cooldown - 1)
        self.shield_timer = max(0, self.shield_timer - 1)
        if self.shield_timer == 0:
            self.shield_active = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_player()
        self._update_projectiles()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Passive Rewards/Penalties ---
        self.temporal_energy = min(100, self.temporal_energy + 0.02) # Slow regeneration
        if self.temporal_energy < 20:
            reward -= 0.01

        # --- Check for Level Completion ---
        if not self.anomalies:
            reward += 100
            self.current_level_index += 1
            if self.current_level_index == 1:
                self.skills_unlocked = True
            if self.current_level_index < self.NUM_LEVELS:
                self._generate_level(self.current_level_index)
            else: # Game Won
                reward += 500
                self.game_over = True

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.temporal_energy <= 0:
            reward -= 50
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
        elif self.game_over: # Win condition
            terminated = True
        
        self.score += reward

        obs = self._get_observation()
        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 3:  # Left
            self.player_vel.x -= 0.8
            self.last_movement_dir = -1
        elif movement == 4:  # Right
            self.player_vel.x += 0.8
            self.last_movement_dir = 1
        
        # Gravity Flip
        if self.flip_cooldown == 0:
            if movement == 1 and self.gravity_dir == 1: # Flip up
                self.gravity_dir = -1
                self.flip_cooldown = 15
                self._spawn_particles(self.player_pos, 30, (200, 200, 255), 1.5, 20, math.pi * 2)
            elif movement == 2 and self.gravity_dir == -1: # Flip down
                self.gravity_dir = 1
                self.flip_cooldown = 15
                self._spawn_particles(self.player_pos, 30, (200, 200, 255), 1.5, 20, math.pi * 2)

        # Shoot
        if space_held and self.shoot_cooldown == 0 and self.temporal_energy > 5:
            self._spawn_projectile()
            self.temporal_energy -= 5
            self.shoot_cooldown = 8 # Fire rate
        
        # Skill: Shield
        if shift_held and self.skills_unlocked and not self.shield_active and self.temporal_energy > 25:
            self.shield_active = True
            self.shield_timer = 90 # 3 seconds at 30fps
            self.temporal_energy -= 25

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += 0.5 * self.gravity_dir
        
        # Apply friction
        self.player_vel.x *= 0.9
        
        # Limit speed
        self.player_vel.x = max(-7, min(7, self.player_vel.x))
        self.player_vel.y = max(-10, min(10, self.player_vel.y))

        # Move and handle platform collisions
        self.player_pos.x += self.player_vel.x
        player_rect = self._get_player_rect()
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.x > 0: player_rect.right = plat.left
                elif self.player_vel.x < 0: player_rect.left = plat.right
                self.player_pos.x = player_rect.centerx
                self.player_vel.x = 0
        
        self.player_pos.y += self.player_vel.y
        player_rect = self._get_player_rect()
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.y > 0: 
                    player_rect.bottom = plat.top
                    self.player_vel.y = 0
                elif self.player_vel.y < 0: 
                    player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = player_rect.centery

        # Screen bounds
        self.player_pos.x = max(10, min(self.WIDTH - 10, self.player_pos.x))
        if (self.player_pos.y > self.HEIGHT + 50) or (self.player_pos.y < -50):
             # Fall out of bounds, reset with penalty
             self.temporal_energy -= 20
             self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
             self.player_vel = pygame.Vector2(0, 0)


    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().colliderect(pygame.Rect(proj['pos'].x, proj['pos'].y, 1, 1)):
                self.projectiles.remove(proj)
            else:
                self._spawn_particles(proj['pos'], 1, self.COLOR_PROJECTILE, 0.5, 10, 0, 0.5)

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0.0
        player_rect = self._get_player_rect()

        # Projectiles vs Anomalies
        for proj in self.projectiles[:]:
            for anomaly in self.anomalies[:]:
                if proj['pos'].distance_to(anomaly['pos']) < anomaly['radius']:
                    reward += 1.0
                    self._spawn_particles(anomaly['pos'], 50, self.COLOR_ANOMALY, 2, 30, math.pi * 2)
                    self._spawn_particles(anomaly['pos'], 20, (255, 255, 255), 1, 20, math.pi * 2)
                    self.anomalies.remove(anomaly)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    break
        
        # Player vs Energy Pickups
        for pickup in self.energy_pickups[:]:
            if player_rect.colliderect(pickup['rect']):
                reward += 0.1
                self.temporal_energy = min(100, self.temporal_energy + 15)
                self._spawn_particles(pickup['rect'].center, 20, (0, 255, 100), 1, 15, math.pi * 2)
                self.energy_pickups.remove(pickup)

        return reward

    def _generate_level(self, level_index):
        self.platforms = []
        self.anomalies = []
        self.energy_pickups = []

        # Add floor and ceiling
        self.platforms.append(pygame.Rect(-10, self.HEIGHT, self.WIDTH + 20, 10))
        self.platforms.append(pygame.Rect(-10, -10, self.WIDTH + 20, 10))

        # Generate platforms and anomalies based on level
        level_seed = level_index
        if self.np_random:
            level_seed = self.np_random.integers(0, 100000) # Use numpy RNG for level generation if available
        
        # Use a local Random instance for deterministic level generation
        level_rng = random.Random(level_seed)

        num_plats = 10 + level_index * 2
        for _ in range(num_plats):
            w = level_rng.randint(80, 200)
            h = 20
            x = level_rng.randint(0, self.WIDTH - w)
            y = level_rng.randint(40, self.HEIGHT - 40)
            self.platforms.append(pygame.Rect(x, y, w, h))

        num_anomalies = 5 + level_index * 2
        for _ in range(num_anomalies):
            while True:
                pos = pygame.Vector2(level_rng.randint(50, self.WIDTH-50), level_rng.randint(50, self.HEIGHT-50))
                # Ensure anomaly is not inside a platform
                is_safe = True
                for plat in self.platforms:
                    if plat.collidepoint(pos):
                        is_safe = False
                        break
                if is_safe:
                    self.anomalies.append({'pos': pos, 'radius': 12, 'angle': level_rng.uniform(0, 2*math.pi)})
                    break
        
        num_pickups = 3
        for _ in range(num_pickups):
             while True:
                pos = (level_rng.randint(50, self.WIDTH-50), level_rng.randint(50, self.HEIGHT-50))
                rect = pygame.Rect(pos[0]-10, pos[1]-10, 20, 20)
                is_safe = True
                for plat in self.platforms:
                    if plat.colliderect(rect):
                        is_safe = False
                        break
                if is_safe:
                    self.energy_pickups.append({'rect': rect, 'bob': level_rng.uniform(0, 2*math.pi)})
                    break

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 15, 16, 30)

    def _spawn_projectile(self):
        start_pos = self.player_pos + pygame.Vector2(self.last_movement_dir * 15, 0)
        velocity = pygame.Vector2(self.last_movement_dir * 15, 0)
        self.projectiles.append({'pos': start_pos, 'vel': velocity})
        self._spawn_particles(start_pos, 10, self.COLOR_PROJECTILE, 1, 10, math.pi * 0.5, angle_offset=math.radians(90 if self.last_movement_dir > 0 else -90))
    
    def _spawn_particles(self, pos, count, color, speed_mult, lifetime, spread=math.pi*2, angle_offset=0):
        for _ in range(count):
            angle = random.uniform(0, spread) - spread/2 + angle_offset
            speed = random.uniform(0.5, 1.5) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            radius = random.uniform(1, 4)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    def _get_observation(self):
        colors = self.SEASON_COLORS[self.current_level_index % self.NUM_LEVELS]
        self.screen.fill(colors["bg"])
        
        self._render_game(colors)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, colors):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, colors["platform"], plat)
            pygame.draw.rect(self.screen, colors["detail"], plat, 2)
        
        # Draw energy pickups
        for pickup in self.energy_pickups:
            pickup['bob'] = (pickup['bob'] + 0.1) % (2 * math.pi)
            y_offset = math.sin(pickup['bob']) * 4
            center = (pickup['rect'].centerx, pickup['rect'].centery + y_offset)
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 8, (0, 255, 100))
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), 8, (200, 255, 220))

        # Draw anomalies
        for anomaly in self.anomalies:
            anomaly['angle'] += 0.05
            radius = anomaly['radius']
            for i in range(3):
                offset_angle = anomaly['angle'] + (i * 2 * math.pi / 3)
                p1 = anomaly['pos'] + pygame.Vector2(math.cos(offset_angle), math.sin(offset_angle)) * radius
                p2 = anomaly['pos'] + pygame.Vector2(math.cos(offset_angle + math.pi), math.sin(offset_angle + math.pi)) * radius * 0.5
                pygame.draw.aaline(self.screen, self.COLOR_ANOMALY, p1, p2, 2)
            pygame.gfxdraw.aacircle(self.screen, int(anomaly['pos'].x), int(anomaly['pos'].y), int(radius * 0.8), self.COLOR_ANOMALY)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, proj['pos'], proj['pos'] - proj['vel'] * 0.5, 3)

        # Draw player
        player_rect = self._get_player_rect()
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        # Glow
        glow_surf = pygame.Surface((40, 60), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (255, 255, 0, 50), glow_surf.get_rect())
        self.screen.blit(glow_surf, (player_rect.centerx - 20, player_rect.centery - 30))
        # Shield
        if self.shield_active:
            alpha = 50 + (self.shield_timer / 90) * 100
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 25, (100, 200, 255, int(alpha/2)))
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), 25, (200, 240, 255, int(alpha)))
    
    def _render_ui(self):
        # Energy Bar
        energy_rect = pygame.Rect(10, 10, 200, 20)
        pygame.draw.rect(self.screen, (50, 50, 50), energy_rect)
        fill_width = max(0, (self.temporal_energy / 100) * energy_rect.width)
        fill_rect = pygame.Rect(10, 10, fill_width, 20)
        pygame.draw.rect(self.screen, (0, 200, 255), fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, energy_rect, 1)
        
        # Anomaly Count
        anomaly_text = self.font_small.render(f"Anomalies: {len(self.anomalies)}", True, self.COLOR_TEXT)
        self.screen.blit(anomaly_text, (self.WIDTH - anomaly_text.get_width() - 10, 10))

        # Level indicator
        level_text = self.font_small.render(f"Season: {self.current_level_index + 1}/{self.NUM_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 35))

        # Game Over/Victory text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self.current_level_index >= self.NUM_LEVELS:
                msg = "SEASONS RESTORED"
            else:
                msg = "TIME COLLAPSED"
            
            end_text = self.font_large.render(msg, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _render_human(self):
        if self.human_screen is None: return
        self.human_screen.blit(self.screen, (0,0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level_index,
            "anomalies_left": len(self.anomalies),
            "energy": self.temporal_energy,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To run in human mode, you might need to remove the SDL_VIDEODRIVER setting
    # For example, run as: SDL_VIDEODRIVER=x11 python your_script_name.py
    
    render_mode = "human"
    if os.getenv("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No window will be displayed.")
        render_mode = "rgb_array"
    
    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard to MultiDiscrete action
    # [Movement, Space, Shift]
    action = [0, 0, 0] 
    
    print("\n--- CONTROLS ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------\n")

    while not done:
        if render_mode == "human":
            keys = pygame.key.get_pressed()
            
            # Reset action
            action = [0, 0, 0] # 0 = No-op for movement

            # Movement (mutually exclusive)
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action[0] = 4
            elif keys[pygame.K_UP] or keys[pygame.K_w]:
                action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action[0] = 2

            # Actions
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
        else: # Headless: step with random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            pass # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            if render_mode != "human": # Stop after one episode in headless mode
                done = True

    env.close()