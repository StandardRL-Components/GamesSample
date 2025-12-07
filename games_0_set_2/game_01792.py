import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Hold Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive against hordes of zombies in a top-down arena shooter. Manage your ammo and health to stay alive as long as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    ZOMBIE_SPAWN_INTERVAL = 500
    ZOMBIE_SPAWN_COUNT = 10
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ZOMBIE = (220, 30, 30)
    COLOR_BULLET = (255, 255, 0)
    COLOR_FLASH = (255, 255, 220)
    COLOR_TEXT = (230, 230, 230)
    COLOR_HEALTH_FG = (50, 200, 50)
    COLOR_HEALTH_BG = (100, 30, 30)
    
    # Player
    PLAYER_SIZE = 16
    PLAYER_SPEED = 5
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 10
    
    # Zombie
    ZOMBIE_SIZE = 14
    ZOMBIE_SPEED = 1.5
    ZOMBIE_HEALTH = 20
    ZOMBIE_DAMAGE = 10
    
    # Bullet
    BULLET_SPEED = 20
    BULLET_DAMAGE = 20
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Set headless mode for Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.np_random = None
        self.player_rect = None
        self.player_health = 0
        self.player_ammo = 0
        self.player_aim_direction = np.array([0.0, -1.0])
        self.is_reloading = False
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.muzzle_flash_timer = 0
        
        # This will be called again in the main block, but is needed for initialization
        # The seed will be properly set by the first call to reset()
        self.reset(seed=0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_rect = pygame.Rect(
            self.WIDTH // 2 - self.PLAYER_SIZE // 2,
            self.HEIGHT // 2 - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_direction = np.array([0.0, -1.0])
        self.is_reloading = False
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self._spawn_zombies(self.ZOMBIE_SPAWN_COUNT)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.muzzle_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Handle Reloading State ---
        if self.is_reloading:
            self.player_ammo = self.PLAYER_MAX_AMMO
            self.is_reloading = False
            # Reloading takes the whole step, no other actions processed
        else:
            self._handle_input(action)

        # --- Update Game State ---
        zombie_kills = self._update_bullets()
        player_hit = self._update_zombies()
        self._update_particles()
        
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1
            
        # --- Spawning ---
        if self.steps > 0 and self.steps % self.ZOMBIE_SPAWN_INTERVAL == 0:
            self._spawn_zombies(self.ZOMBIE_SPAWN_COUNT)
            
        # --- Calculate Rewards ---
        reward += 0.01  # Small reward for surviving a step
        reward += zombie_kills * 1.0 # Reward for killing zombies
        if player_hit:
            self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
            self._create_particles(self.player_rect.center, self.COLOR_PLAYER, 20, 2.0)
            # Negative reward for getting hit is implicit in losing the game

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            self.win = False
        elif self.steps >= self.MAX_STEPS:
            reward = 100.0
            truncated = True # Truncated, not terminated, for reaching step limit
            self.game_over = True
            self.win = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement and Aiming
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1 # Up
        elif movement == 2: move_vec[1] += 1 # Down
        elif movement == 3: move_vec[0] -= 1 # Left
        elif movement == 4: move_vec[0] += 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player_aim_direction = move_vec
            self.player_rect.x += move_vec[0] * self.PLAYER_SPEED
            self.player_rect.y += move_vec[1] * self.PLAYER_SPEED

        self.player_rect.clamp_ip(self.screen.get_rect())

        # Actions
        if shift_held and self.player_ammo < self.PLAYER_MAX_AMMO:
            self.is_reloading = True
        elif space_held and self.player_ammo > 0:
            self._shoot()
            
    def _shoot(self):
        self.player_ammo -= 1
        self.muzzle_flash_timer = 2 # frames
        
        start_pos = np.array(self.player_rect.center, dtype=float)
        bullet_vel = self.player_aim_direction * self.BULLET_SPEED
        
        bullet = {
            "rect": pygame.Rect(start_pos[0]-2, start_pos[1]-2, 4, 4),
            "vel": bullet_vel,
            "pos": start_pos
        }
        self.bullets.append(bullet)

    def _spawn_zombies(self, count):
        for _ in range(count):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                x, y = self.np_random.integers(self.WIDTH), -self.ZOMBIE_SIZE
            elif edge == 1: # Bottom
                x, y = self.np_random.integers(self.WIDTH), self.HEIGHT
            elif edge == 2: # Left
                x, y = -self.ZOMBIE_SIZE, self.np_random.integers(self.HEIGHT)
            else: # Right
                x, y = self.WIDTH, self.np_random.integers(self.HEIGHT)
            
            zombie = {
                "rect": pygame.Rect(x, y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE),
                "health": self.ZOMBIE_HEALTH,
                "move_pattern_timer": self.np_random.integers(30, 90),
                "move_type": 0 # 0=direct, 1=sideways
            }
            self.zombies.append(zombie)
            
    def _update_bullets(self):
        kills = 0
        for b in self.bullets[:]:
            b["pos"] += b["vel"]
            b["rect"].center = b["pos"]
            
            # Check for zombie collision
            hit_zombie = False
            for z in self.zombies[:]:
                if b["rect"].colliderect(z["rect"]):
                    z["health"] -= self.BULLET_DAMAGE
                    if z["health"] <= 0:
                        self._create_particles(z["rect"].center, self.COLOR_ZOMBIE, 30, 3.0)
                        self.zombies.remove(z)
                        kills += 1
                    hit_zombie = True
                    break
            
            if hit_zombie or not self.screen.get_rect().colliderect(b["rect"]):
                if b in self.bullets:
                    self.bullets.remove(b)
        return kills

    def _update_zombies(self):
        player_hit = False
        player_center = np.array(self.player_rect.center, dtype=float)
        
        for z in self.zombies[:]:
            zombie_center = np.array(z["rect"].center)
            direction = player_center - zombie_center
            dist = np.linalg.norm(direction)
            
            if dist > 1:
                direction /= dist # Normalize
            
            # Zig-zag movement pattern
            z["move_pattern_timer"] -= 1
            if z["move_pattern_timer"] <= 0:
                z["move_type"] = self.np_random.integers(2)
                z["move_pattern_timer"] = self.np_random.integers(30, 90)

            move_vec = direction
            if z["move_type"] == 1: # Sideways movement
                move_vec = np.array([-direction[1], direction[0]])
            
            z["rect"].x += move_vec[0] * self.ZOMBIE_SPEED
            z["rect"].y += move_vec[1] * self.ZOMBIE_SPEED
            
            # Check player collision
            if z["rect"].colliderect(self.player_rect):
                player_hit = True
                if z in self.zombies:
                    self.zombies.remove(z)
        return player_hit

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        # pygame.surfarray.array3d returns (width, height, 3), we need (height, width, 3)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            size = max(1, int(p["lifespan"] / 6))
            pygame.draw.circle(self.screen, p["color"], [int(p["pos"][0]), int(p["pos"][1])], size)

        # Bullets
        for b in self.bullets:
            end_pos = b["pos"] + b["vel"] * 0.5
            pygame.draw.line(self.screen, self.COLOR_BULLET, b["rect"].center, (int(end_pos[0]), int(end_pos[1])), 2)

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        # Player glow
        glow_rect = self.player_rect.inflate(8, 8)
        try:
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_PLAYER, 30))
        except TypeError: # Older pygame versions might not support alpha
            pass
        
        # Muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player_rect.center + self.player_aim_direction * 12
            radius = 10 + self.np_random.integers(-2, 3)
            pygame.draw.circle(self.screen, self.COLOR_FLASH, (int(flash_pos[0]), int(flash_pos[1])), radius)

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, health_bar_width * health_ratio, 20))
        
        # Ammo Count
        ammo_text = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        if self.is_reloading:
            ammo_text = "RELOADING..."
        text_surf = self.font_small.render(ammo_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU SURVIVED" if self.win else "YOU DIED"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies),
        }
        
    def close(self):
        pygame.quit()

# Example usage to test the environment
if __name__ == '__main__':
    # The environment is validated in the local test harness, no need for a separate validation method
    
    # Run headless for testing
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    print("Initial State:")
    print(f"  Info: {info}")

    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        if terminated or truncated:
            print("Episode finished.")
            break
            
    env.close()

    # To visualize the game (requires a display)
    # Set the environment variable to your display driver, e.g., "x11", "wayland", "windows"
    # unset SDL_VIDEODRIVER or set it to a valid driver
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        # Re-initialize pygame with a display
        pygame.quit()
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or your specific driver
        pygame.init()
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        
        # Create a display window
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            # Map keyboard to actions for human play
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                pygame.time.wait(3000) # Wait 3 seconds before resetting
                obs, info = env.reset(seed=43) # Use a new seed for a new game

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(30) # Limit FPS for human play
            
        env.close()