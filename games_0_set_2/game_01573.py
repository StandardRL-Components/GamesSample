
# Generated: 2025-08-28T02:01:11.555840
# Source Brief: brief_01573.md
# Brief Index: 1573

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Press space to shoot. Press shift to reload."
    )

    # User-facing description of the game
    game_description = (
        "Survive waves of procedurally generated zombies in a top-down arena shooter."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (100, 100, 110)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_ZOMBIE_GLOW = (255, 50, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_PROJECTILE_GLOW = (255, 255, 0, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_AMMO_BAR = (200, 200, 50)
        self.COLOR_BAR_BG = (70, 70, 70)
        self.COLOR_RELOAD = (255, 150, 0)
        self.COLOR_SCREEN_FLASH = (255, 0, 0, 100)

        # Player settings
        self.PLAYER_SPEED = 4
        self.PLAYER_RADIUS = 12
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 30
        self.PLAYER_SHOOT_COOLDOWN = 5  # frames
        self.PLAYER_RELOAD_TIME = 60 # frames (2 seconds at 30fps)
        self.PLAYER_DAMAGE_COOLDOWN = 30 # frames of invincibility after hit

        # Zombie settings
        self.ZOMBIE_RADIUS = 10
        self.ZOMBIE_BASE_SPEED = 1.0
        self.ZOMBIE_SPEED_INCREMENT = 0.05
        self.ZOMBIE_DAMAGE = 10

        # Projectile settings
        self.PROJECTILE_SPEED = 12
        self.PROJECTILE_RADIUS = 3

        # Wave settings
        self.INITIAL_ZOMBIE_COUNT = 10
        self.ZOMBIE_INCREMENT_PER_WAVE = 5
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_vector = None
        self.shoot_cooldown = None
        self.reloading = None
        self.reload_timer = None
        self.damage_cooldown = None
        self.screen_flash_timer = None
        
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.wave = None
        self.zombie_speed = None
        
        self.steps = 0
        self.score = 0
        self.step_reward = 0
        self.game_over = False

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_vector = pygame.math.Vector2(0, -1) # Default aim up
        
        self.shoot_cooldown = 0
        self.reloading = False
        self.reload_timer = 0
        self.damage_cooldown = 0
        self.screen_flash_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.wave = 1
        self.zombie_speed = self.ZOMBIE_BASE_SPEED
        self._spawn_zombies()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.step_reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement, space_held, shift_held)
        self._update_zombies()
        self._update_projectiles()
        self._update_particles()
        
        self._handle_collisions()
        
        self._check_wave_completion()

        terminated = self._check_termination()
        
        reward = self.step_reward
        if terminated and self.player_health <= 0:
            reward = -100 # Large penalty for dying

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held, shift_held):
        # Cooldowns
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.damage_cooldown > 0: self.damage_cooldown -= 1
        if self.screen_flash_timer > 0: self.screen_flash_timer -= 1
        
        # Movement and Aiming
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1
        elif movement == 2: move_vector.y = 1
        elif movement == 3: move_vector.x = -1
        elif movement == 4: move_vector.x = 1
        
        if move_vector.length() > 0:
            self.player_aim_vector = move_vector.copy()
            # Normalize to prevent faster diagonal movement
            move_vector.normalize_ip()
            self.player_pos += move_vector * self.PLAYER_SPEED

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # Reloading
        if self.reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.reloading = False
                self.player_ammo = self.PLAYER_MAX_AMMO
                # Reload complete sound
        elif shift_held and self.player_ammo < self.PLAYER_MAX_AMMO:
            self.reloading = True
            self.reload_timer = self.PLAYER_RELOAD_TIME
            # Reload start sound

        # Shooting
        if space_held and self.player_ammo > 0 and self.shoot_cooldown == 0 and not self.reloading:
            self._shoot()

    def _shoot(self):
        self.player_ammo -= 1
        self.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
        
        # Projectile starts slightly in front of the player
        start_pos = self.player_pos + self.player_aim_vector * (self.PLAYER_RADIUS + 1)
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": self.player_aim_vector * self.PROJECTILE_SPEED,
        })
        # Gunshot sound
        
        # Muzzle flash particles
        for _ in range(10):
            angle = self.player_aim_vector.angle_to(pygame.math.Vector2(1,0)) + random.uniform(-20, 20)
            speed = random.uniform(2, 5)
            vel = pygame.math.Vector2()
            vel.from_polar((speed, -angle))
            self.particles.append({
                "pos": start_pos.copy(),
                "vel": vel,
                "radius": random.uniform(1, 4),
                "color": random.choice([(255,255,0), (255,150,0)]),
                "lifespan": random.randint(5, 15)
            })

    def _update_zombies(self):
        for z in self.zombies:
            direction = (self.player_pos - z["pos"]).normalize()
            z["pos"] += direction * self.zombie_speed
            
    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)
                self.step_reward -= 0.01 # Penalty for missing

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] -= 0.1
            if p["lifespan"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            for z in self.zombies[:]:
                if (p["pos"] - z["pos"]).length() < self.PROJECTILE_RADIUS + self.ZOMBIE_RADIUS:
                    self.projectiles.remove(p)
                    self.zombies.remove(z)
                    self.step_reward += 1.1 # +0.1 for hit, +1.0 for kill
                    self.score += 10
                    self._create_explosion(z["pos"])
                    # Zombie death sound
                    break # Move to next projectile

        # Player vs Zombies
        if self.damage_cooldown == 0:
            for z in self.zombies:
                if (self.player_pos - z["pos"]).length() < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.damage_cooldown = self.PLAYER_DAMAGE_COOLDOWN
                    self.screen_flash_timer = 5
                    self.score -= 5 # Small score penalty for getting hit
                    # Player hurt sound
                    if self.player_health <= 0:
                        self.game_over = True
                        self._create_explosion(self.player_pos, 2.0)
                        # Player death sound
                    break

    def _create_explosion(self, pos, scale=1.0):
        num_particles = int(30 * scale)
        for _ in range(num_particles):
            speed = random.uniform(1, 6) * scale
            angle = random.uniform(0, 360)
            vel = pygame.math.Vector2()
            vel.from_polar((speed, angle))
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.uniform(2, 5) * scale,
                "color": random.choice([(255, 50, 50), (255, 150, 0), (255, 255, 0)]),
                "lifespan": random.randint(15, 30)
            })

    def _spawn_zombies(self):
        zombie_count = self.INITIAL_ZOMBIE_COUNT + (self.wave - 1) * self.ZOMBIE_INCREMENT_PER_WAVE
        for _ in range(zombie_count):
            while True:
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top': pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), -self.ZOMBIE_RADIUS)
                elif edge == 'bottom': pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_RADIUS)
                elif edge == 'left': pos = pygame.math.Vector2(-self.ZOMBIE_RADIUS, random.uniform(0, self.HEIGHT))
                else: pos = pygame.math.Vector2(self.WIDTH + self.ZOMBIE_RADIUS, random.uniform(0, self.HEIGHT))
                
                # Ensure zombies don't spawn too close to the player
                if (pos - self.player_pos).length() > 150:
                    self.zombies.append({"pos": pos})
                    break

    def _check_wave_completion(self):
        if not self.zombies and not self.game_over:
            self.wave += 1
            self.zombie_speed += self.ZOMBIE_SPEED_INCREMENT
            self._spawn_zombies()
            
            # Reward for clearing wave, scaled by remaining health
            wave_clear_bonus = 10 + 90 * (self.player_health / self.PLAYER_MAX_HEALTH)
            self.step_reward += wave_clear_bonus
            self.score += 100
            # Wave clear sound

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render projectiles
        for p in self.projectiles:
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS + 2, self.COLOR_PROJECTILE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Render zombies
        for z in self.zombies:
            pos_int = (int(z["pos"].x), int(z["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_RADIUS + 3, self.COLOR_ZOMBIE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)

        # Render player
        if self.player_health > 0:
            pos_int = (int(self.player_pos.x), int(self.player_pos.y))
            
            # Invincibility flash
            player_color = self.COLOR_PLAYER
            if self.damage_cooldown > 0 and (self.steps // 3) % 2 == 0:
                player_color = self.COLOR_BG

            # Player glow and body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, player_color)
            
            # Aiming indicator
            end_pos = self.player_pos + self.player_aim_vector * self.PLAYER_RADIUS
            pygame.draw.line(self.screen, self.COLOR_BG, pos_int, (int(end_pos.x), int(end_pos.y)), 3)
            
        # Render particles
        for p in self.particles:
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, p["color"])

        # Render screen flash on damage
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_SCREEN_FLASH)
            self.screen.blit(flash_surface, (0,0))

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 10, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, 150 * health_ratio, 15))

        # Ammo bar
        ammo_ratio = max(0, self.player_ammo / self.PLAYER_MAX_AMMO)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 30, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR, (10, 30, 150 * ammo_ratio, 15))

        # Reloading text
        if self.reloading:
            reload_text = self.font_small.render("RELOADING...", True, self.COLOR_RELOAD)
            self.screen.blit(reload_text, (10, 50))

        # Wave text
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Score text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))

        # Game Over text
        if self.game_over:
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_ZOMBIE)
            self.screen.blit(game_over_text, (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 2 - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Zombie Arena")
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to our action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # Action defaults
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        
        # Determine movement action (only one at a time)
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize UP > DOWN > LEFT > RIGHT if multiple are pressed
        
        # Determine button actions
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the Pygame window
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_transposed = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_transposed)
        pygame_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for 'R' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")
                        waiting_for_reset = False

        env.clock.tick(env.FPS)
        
    env.close()