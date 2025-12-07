import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:00:42.001024
# Source Brief: brief_01464.md
# Brief Index: 1464
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent defends a bioluminescent coral reef
    from invasive starfish. The agent controls a central coral polyp, switching
    between attack and defense modes, aiming and firing projectiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a bioluminescent coral reef from invasive starfish. Switch between attack and defense modes to fire projectiles and protect the central coral."
    )
    user_guide = (
        "Controls: Use arrow keys or WASD to aim. Press space to fire projectiles (Attack mode only). Press shift to toggle between Attack and Defense modes."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # === Pygame Setup ===
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_mode = pygame.font.SysFont("Consolas", 24, bold=True)

        # === Colors ===
        self.COLOR_BG_TOP = (5, 10, 25)
        self.COLOR_BG_BOTTOM = (10, 5, 15)
        self.COLOR_CORAL_HEALTHY = (0, 255, 150)
        self.COLOR_CORAL_HURT = (255, 100, 50)
        self.COLOR_AURA_DEFENSE = (0, 150, 255)
        self.COLOR_AURA_ATTACK = (255, 150, 0)
        self.COLOR_STARFISH = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_RETICLE = (255, 255, 255, 150)

        # === Game Parameters ===
        self.MAX_STEPS = 1000
        self.PLAYER_POS = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_RADIUS = 15
        self.DEFENSE_DAMAGE_MULTIPLIER = 0.5
        
        self.AIM_SENSITIVITY = 0.05
        self.AIM_RADIUS = 100

        self.PROJECTILE_SPEED = 7.0
        self.PROJECTILE_RADIUS = 5
        self.FIRE_COOLDOWN_MAX = 10  # 3 shots per second at 30fps

        self.STARFISH_RADIUS = 12
        self.STARFISH_DAMAGE = 20
        self.INITIAL_SPAWN_PROB = 0.01 # 1 per 100 steps
        self.MAX_SPAWN_PROB = 0.1
        self.SPAWN_PROB_INCREASE = 0.0001 # a bit slower than brief for better curve
        self.INITIAL_SPEED = 0.5
        self.MAX_SPEED = 2.0
        self.SPEED_INCREASE_INTERVAL = 100
        self.SPEED_INCREASE_AMOUNT = 0.1 # a bit slower than brief

        # === State Variables (initialized in reset) ===
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.player_mode = 0  # 0: Defense, 1: Attack
        self.aim_angle = 0.0
        self.fire_cooldown = 0
        self.last_shift_state = 0
        
        self.starfish = []
        self.projectiles = []
        self.particles = []
        
        self.starfish_spawn_prob = 0.0
        self.starfish_base_speed = 0.0
        
        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_mode = 0  # Start in defense
        self.aim_angle = -math.pi / 2  # Start aiming straight up
        self.fire_cooldown = 0
        self.last_shift_state = 0

        self.starfish.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.starfish_spawn_prob = self.INITIAL_SPAWN_PROB
        self.starfish_base_speed = self.INITIAL_SPEED
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # === Action Processing ===
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # 1. Aiming (Movement)
        if movement == 1: self.aim_angle -= self.AIM_SENSITIVITY # Up
        if movement == 2: self.aim_angle += self.AIM_SENSITIVITY # Down
        if movement == 3: self.aim_angle -= self.AIM_SENSITIVITY # Left
        if movement == 4: self.aim_angle += self.AIM_SENSITIVITY # Right
        # Clamp angle to top hemisphere
        self.aim_angle = max(-math.pi, min(0, self.aim_angle))

        # 2. Mode Switching (Shift) - Toggle on press
        if shift_held and not self.last_shift_state:
            self.player_mode = 1 - self.player_mode # Toggle 0 and 1
            # sfx: mode_switch.wav
        self.last_shift_state = shift_held
        
        # 3. Firing (Space)
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        if space_held and self.player_mode == 1 and self.fire_cooldown == 0:
            self._fire_projectile()
            self.fire_cooldown = self.FIRE_COOLDOWN_MAX
            # sfx: fire_projectile.wav

        # === Game Logic Updates ===
        reward += self._update_projectiles()
        reward += self._update_starfish()
        self._update_particles()
        self._spawn_starfish()
        self._update_difficulty()

        # === Termination Check ===
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100
            truncated = True # Use truncated for time limit
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _fire_projectile(self):
        direction = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)], dtype=np.float32)
        velocity = direction * self.PROJECTILE_SPEED
        self.projectiles.append({
            "pos": self.PLAYER_POS.copy(),
            "vel": velocity
        })

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            
            # Off-screen check
            if not (0 < proj["pos"][0] < self.WIDTH and 0 < proj["pos"][1] < self.HEIGHT):
                self.projectiles.remove(proj)
                continue
            
            # Collision with starfish
            for star in self.starfish[:]:
                dist = np.linalg.norm(proj["pos"] - star["pos"])
                if dist < self.PROJECTILE_RADIUS + self.STARFISH_RADIUS:
                    # sfx: starfish_hit.wav
                    self._create_particles(star["pos"], self.COLOR_STARFISH, 20, 3)
                    self.starfish.remove(star)
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    self.score += 10
                    reward += 1.0 # Kill reward
                    reward += 0.1 # Hit reward
                    break
        return reward

    def _update_starfish(self):
        reward = 0
        for star in self.starfish[:]:
            direction = (self.PLAYER_POS - star["pos"])
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            
            # Add lateral wiggle for more natural movement
            wiggle = np.array([math.sin(self.steps * 0.1 + star["phase"]), math.cos(self.steps * 0.1 + star["phase"])]) * 0.2
            final_dir = direction + wiggle
            if np.linalg.norm(final_dir) > 0:
                final_dir /= np.linalg.norm(final_dir)

            star["pos"] += final_dir * star["speed"]

            # Collision with player coral
            if np.linalg.norm(star["pos"] - self.PLAYER_POS) < self.STARFISH_RADIUS + self.PLAYER_RADIUS:
                # sfx: coral_damage.wav
                damage = self.STARFISH_DAMAGE
                if self.player_mode == 0: # Defense mode
                    damage *= self.DEFENSE_DAMAGE_MULTIPLIER
                
                self.player_health -= damage
                reward -= 0.01 * damage
                
                self._create_particles(self.PLAYER_POS, self.COLOR_CORAL_HURT, 30, 4)
                self.starfish.remove(star)
        return reward

    def _spawn_starfish(self):
        if self.np_random.random() < self.starfish_spawn_prob:
            side = self.np_random.integers(0, 3) # 0: top, 1: left, 2: right
            if side == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.STARFISH_RADIUS], dtype=np.float32)
            elif side == 1: # Left
                pos = np.array([-self.STARFISH_RADIUS, self.np_random.uniform(0, self.HEIGHT * 0.7)], dtype=np.float32)
            else: # Right
                pos = np.array([self.WIDTH + self.STARFISH_RADIUS, self.np_random.uniform(0, self.HEIGHT * 0.7)], dtype=np.float32)
            
            speed = self.np_random.uniform(self.starfish_base_speed * 0.8, self.starfish_base_speed * 1.2)
            
            self.starfish.append({
                "pos": pos,
                "speed": speed,
                "phase": self.np_random.uniform(0, 2 * math.pi) # For wiggle
            })

    def _update_difficulty(self):
        # Increase spawn probability
        self.starfish_spawn_prob = min(self.MAX_SPAWN_PROB, self.starfish_spawn_prob + self.SPAWN_PROB_INCREASE)
        # Increase speed
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.starfish_base_speed = min(self.MAX_SPEED, self.starfish_base_speed + self.SPEED_INCREASE_AMOUNT)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_strength=3):
        x, y = int(pos[0]), int(pos[1])
        for i in range(glow_strength, 0, -1):
            alpha = int(100 / (i**1.5))
            pygame.gfxdraw.aacircle(surface, x, y, int(radius + i * 2), (*color, alpha))
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)

    def _get_observation(self):
        # --- Background ---
        # A simple gradient for the deep sea effect
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Game Elements ---
        self._render_game()
        
        # --- UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = p["lifespan"] * 8
            color = (*p["color"], max(0, min(255, alpha)))
            pygame.draw.circle(self.screen, color, p["pos"], 2)

        # Render projectiles
        for proj in self.projectiles:
            self._draw_glow_circle(self.screen, proj["pos"], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE, glow_strength=2)
            
        # Render starfish
        for star in self.starfish:
            self._draw_glow_circle(self.screen, star["pos"], self.STARFISH_RADIUS, self.COLOR_STARFISH, glow_strength=3)

        # Render player coral
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        coral_color = (
            int(self.COLOR_CORAL_HURT[0] + (self.COLOR_CORAL_HEALTHY[0] - self.COLOR_CORAL_HURT[0]) * health_ratio),
            int(self.COLOR_CORAL_HURT[1] + (self.COLOR_CORAL_HEALTHY[1] - self.COLOR_CORAL_HURT[1]) * health_ratio),
            int(self.COLOR_CORAL_HURT[2] + (self.COLOR_CORAL_HEALTHY[2] - self.COLOR_CORAL_HURT[2]) * health_ratio)
        )
        aura_color = self.COLOR_AURA_ATTACK if self.player_mode == 1 else self.COLOR_AURA_DEFENSE
        
        # Pulsing aura
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # Varies between 0 and 1
        aura_radius = self.PLAYER_RADIUS + 10 + pulse * 5
        self._draw_glow_circle(self.screen, self.PLAYER_POS, aura_radius, aura_color, glow_strength=5)
        self._draw_glow_circle(self.screen, self.PLAYER_POS, self.PLAYER_RADIUS, coral_color, glow_strength=4)

        # Render aiming reticle
        if self.player_mode == 1:
            reticle_x = self.PLAYER_POS[0] + math.cos(self.aim_angle) * self.AIM_RADIUS
            reticle_y = self.PLAYER_POS[1] + math.sin(self.aim_angle) * self.AIM_RADIUS
            pygame.gfxdraw.aacircle(self.screen, int(reticle_x), int(reticle_y), 8, self.COLOR_RETICLE)
            pygame.gfxdraw.line(self.screen, int(reticle_x) - 5, int(reticle_y), int(reticle_x) + 5, int(reticle_y), self.COLOR_RETICLE)
            pygame.gfxdraw.line(self.screen, int(reticle_x), int(reticle_y) - 5, int(reticle_x), int(reticle_y) + 5, self.COLOR_RETICLE)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps remaining
        steps_rem = self.MAX_STEPS - self.steps
        steps_text = self.font_ui.render(f"TIME: {steps_rem}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Mode display
        mode_str = "ATTACK" if self.player_mode == 1 else "DEFENSE"
        mode_color = self.COLOR_AURA_ATTACK if self.player_mode == 1 else self.COLOR_AURA_DEFENSE
        mode_text = self.font_mode.render(mode_str, True, mode_color)
        self.screen.blit(mode_text, (self.WIDTH // 2 - mode_text.get_width() // 2, self.HEIGHT - 35))

        # Health bar
        health_bar_width = 100
        health_bar_height = 10
        health_bar_x = self.PLAYER_POS[0] - health_bar_width // 2
        health_bar_y = self.PLAYER_POS[1] + self.PLAYER_RADIUS + 15
        
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        current_health_width = health_bar_width * health_ratio
        
        pygame.draw.rect(self.screen, (50, 50, 50), (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_CORAL_HEALTHY, (health_bar_x, health_bar_y, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "player_mode": "attack" if self.player_mode == 1 else "defense"
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment interface.
    # It will not be executed by the test suite.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Coral Guardian")
    clock = pygame.time.Clock()

    # Action state
    action = [0, 0, 0] # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Keyboard controls for human play
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op default
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth human play

    print(f"Game Over! Final Info: {info}")
    env.close()