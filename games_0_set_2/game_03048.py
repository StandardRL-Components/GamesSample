
# Generated: 2025-08-27T22:13:38.531331
# Source Brief: brief_03048.md
# Brief Index: 3048

        
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
        "Controls: ←→ to move, ↑ to jump. Hold space to shoot. Survive the horde."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling zombie survival shooter. Last for 5 minutes against an ever-growing horde."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 10  # Simulation steps per second
        self.MAX_STEPS = 3000  # 5 minutes at 10 FPS

        # Colors
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GROUND = (40, 35, 50)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_ZOMBIE = (100, 150, 80)
        self.COLOR_ZOMBIE_BLOOD = (120, 180, 100)
        self.COLOR_BULLET = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH = (255, 50, 50)
        self.COLOR_AMMO = (255, 220, 100)
        self.COLOR_TIMER = (200, 200, 255)
        self.COLOR_MUZZLE_FLASH = (255, 255, 200)

        # Player settings
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_AMMO_MAX = 50
        self.PLAYER_SPEED = 5
        self.PLAYER_JUMP_STRENGTH = 14
        self.PLAYER_GRAVITY = 1.0
        self.PLAYER_FRICTION = 0.8
        self.PLAYER_SHOOT_COOLDOWN = 3  # 3 steps

        # Zombie settings
        self.ZOMBIE_HEALTH_MAX = 10
        self.ZOMBIE_SPEED_MIN, self.ZOMBIE_SPEED_MAX = 1, 3
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_INITIAL_SPAWN_RATE = 2.0  # seconds
        self.ZOMBIE_SPAWN_RATE_INCREASE = 0.001 # per second

        # World settings
        self.GROUND_HEIGHT = 60

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_ammo = None
        self.player_is_grounded = None
        self.player_facing_right = None
        self.player_shoot_cooldown_timer = None
        self.player_damage_timer = None
        
        self.zombies = None
        self.bullets = None
        self.particles = None
        
        self.zombie_spawn_timer = None
        self.zombie_spawn_rate = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.muzzle_flash_timer = None

        self.bg_buildings_1 = []
        self.bg_buildings_2 = []
        self.camera_offset = 0.0

        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = [self.WIDTH // 4, self.HEIGHT - self.GROUND_HEIGHT]
        self.player_vel = [0, 0]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = self.PLAYER_AMMO_MAX
        self.player_is_grounded = True
        self.player_facing_right = True
        self.player_shoot_cooldown_timer = 0
        self.player_damage_timer = 0

        # Game state
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.zombie_spawn_rate = self.ZOMBIE_INITIAL_SPAWN_RATE * self.FPS
        self.zombie_spawn_timer = self.zombie_spawn_rate

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.muzzle_flash_timer = 0
        
        self.camera_offset = 0.0
        self._generate_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Survival penalty

        # --- ACTION HANDLING ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -self.PLAYER_SPEED
            self.player_facing_right = False
        elif movement == 4: # Right
            target_vx = self.PLAYER_SPEED
            self.player_facing_right = True
        
        self.player_vel[0] = target_vx
        
        # Jumping
        if movement == 1 and self.player_is_grounded: # Up
            self.player_vel[1] = -self.PLAYER_JUMP_STRENGTH
            self.player_is_grounded = False
            # sfx: player_jump

        # Shooting
        if space_held and self.player_shoot_cooldown_timer == 0 and self.player_ammo > 0:
            self._spawn_bullet()
            self.player_ammo -= 1
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
            self.muzzle_flash_timer = 2 # frames
            # sfx: player_shoot

        # --- GAME LOGIC UPDATE ---
        self._update_player()
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        # Cooldowns and timers
        if self.player_shoot_cooldown_timer > 0: self.player_shoot_cooldown_timer -= 1
        if self.player_damage_timer > 0: self.player_damage_timer -= 1
        if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
        
        # --- COLLISIONS ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- ZOMBIE SPAWNING ---
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            self._spawn_zombie()
            # Increase spawn rate
            rate_decrease = (self.ZOMBIE_SPAWN_RATE_INCREASE / self.FPS) * self.FPS
            self.zombie_spawn_rate = max(0.5 * self.FPS, self.zombie_spawn_rate - rate_decrease)
            self.zombie_spawn_timer = self.zombie_spawn_rate

        # --- STATE UPDATE ---
        self.steps += 1
        self.score += reward

        # --- TERMINATION ---
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            # sfx: game_over_lose
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            self.win = True
            reward += 100  # Survival bonus
            self.score += 100
            # sfx: game_over_win

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _spawn_bullet(self):
        direction = 1 if self.player_facing_right else -1
        # Position bullet at player's "gun"
        start_pos = [self.player_pos[0] + 20 * direction, self.player_pos[1] - 15]
        self.bullets.append({
            "pos": start_pos,
            "vel": [20 * direction, 0],
            "rect": pygame.Rect(start_pos[0], start_pos[1], 8, 4)
        })

    def _spawn_zombie(self):
        side = self.WIDTH + 20
        pos = [side, self.HEIGHT - self.GROUND_HEIGHT]
        speed = self.np_random.uniform(self.ZOMBIE_SPEED_MIN, self.ZOMBIE_SPEED_MAX)
        self.zombies.append({
            "pos": pos,
            "vel": [-speed, 0],
            "health": self.ZOMBIE_HEALTH_MAX,
            "rect": pygame.Rect(pos[0] - 10, pos[1] - 30, 20, 30),
            "walk_phase": self.np_random.uniform(0, 2 * math.pi)
        })

    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.PLAYER_GRAVITY
        
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # Ground collision
        if self.player_pos[1] >= self.HEIGHT - self.GROUND_HEIGHT:
            self.player_pos[1] = self.HEIGHT - self.GROUND_HEIGHT
            self.player_vel[1] = 0
            self.player_is_grounded = True
        
        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        
        # Update camera offset for parallax
        self.camera_offset += (self.player_pos[0] - self.WIDTH // 4 - self.camera_offset) * 0.1

    def _update_bullets(self):
        for b in self.bullets:
            b["pos"][0] += b["vel"][0]
            b["pos"][1] += b["vel"][1]
            b["rect"].topleft = b["pos"]
        self.bullets = [b for b in self.bullets if 0 < b["pos"][0] < self.WIDTH]

    def _update_zombies(self):
        for z in self.zombies:
            z["pos"][0] += z["vel"][0]
            z["pos"][1] += z["vel"][1]
            z["rect"].midbottom = z["pos"]
        self.zombies = [z for z in self.zombies if z["pos"][0] > -50]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # particle gravity
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Zombies
        for b in self.bullets[:]:
            for z in self.zombies[:]:
                if b["rect"].colliderect(z["rect"]):
                    self.bullets.remove(b)
                    z["health"] -= 10 # 1-shot kill
                    # sfx: zombie_hit
                    self._create_particles(z["rect"].center, 15, self.COLOR_ZOMBIE_BLOOD, 4, 30)
                    if z["health"] <= 0:
                        self.zombies.remove(z)
                        reward += 1.0
                        # sfx: zombie_death
                    break
        
        # Player vs Zombies
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 30, 20, 30)
        if self.player_damage_timer == 0:
            for z in self.zombies:
                if player_rect.colliderect(z["rect"]):
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.player_damage_timer = 2 * self.FPS # 2 second immunity
                    reward -= 1.0
                    # sfx: player_damage
                    self._create_particles(player_rect.center, 10, self.COLOR_HEALTH, 3, 20)
                    # Knockback
                    knockback_dir = 1 if player_rect.centerx < z["rect"].centerx else -1
                    self.player_vel[0] = -knockback_dir * 5
                    self.player_vel[1] = -5
                    break
        
        self.player_health = max(0, self.player_health)
        return reward

    def _create_particles(self, pos, count, color, speed, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            p_speed = self.np_random.uniform(0.5, speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * p_speed, math.sin(angle) * p_speed],
                "lifetime": self.np_random.integers(lifetime // 2, lifetime),
                "color": color
            })
            
    def _generate_background(self):
        self.bg_buildings_1 = []
        for i in range(20):
            w = self.np_random.integers(40, 100)
            h = self.np_random.integers(50, 200)
            x = i * 150 + self.np_random.integers(-30, 30)
            self.bg_buildings_1.append(pygame.Rect(x, self.HEIGHT - self.GROUND_HEIGHT - h, w, h))
        
        self.bg_buildings_2 = []
        for i in range(30):
            w = self.np_random.integers(30, 80)
            h = self.np_random.integers(30, 150)
            x = i * 100 + self.np_random.integers(-20, 20)
            self.bg_buildings_2.append(pygame.Rect(x, self.HEIGHT - self.GROUND_HEIGHT - h, w, h))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render background
        self._render_background()
        
        # Render ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - self.GROUND_HEIGHT, self.WIDTH, self.GROUND_HEIGHT))

        # Render game elements
        self._render_particles()
        self._render_zombies()
        self._render_player()
        self._render_bullets()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Far buildings (darker, slower scroll)
        for b in self.bg_buildings_2:
            rect = b.copy()
            rect.x -= int(self.camera_offset * 0.2)
            pygame.draw.rect(self.screen, (35, 30, 45), rect)
        # Near buildings (lighter, faster scroll)
        for b in self.bg_buildings_1:
            rect = b.copy()
            rect.x -= int(self.camera_offset * 0.5)
            pygame.draw.rect(self.screen, (45, 40, 55), rect)
            
    def _render_player(self):
        if self.player_health <= 0: return

        # Player color flashes when damaged
        player_color = self.COLOR_PLAYER
        if self.player_damage_timer > 0 and (self.steps % 4 < 2):
            player_color = (255, 255, 255)

        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Simple sprite: body, head, gun
        body_rect = pygame.Rect(px - 8, py - 30, 16, 20)
        head_pos = (px, py - 35)
        gun_dir = 1 if self.player_facing_right else -1
        gun_rect = pygame.Rect(px - 4 + (gun_dir*4), py - 20, 16 * gun_dir, 5)

        pygame.draw.rect(self.screen, player_color, body_rect, border_radius=3)
        pygame.gfxdraw.filled_circle(self.screen, head_pos[0], head_pos[1], 7, player_color)
        pygame.draw.rect(self.screen, player_color, gun_rect, border_radius=2)
        
        # Muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = (gun_rect.right if self.player_facing_right else gun_rect.left, gun_rect.centery)
            pygame.gfxdraw.filled_circle(self.screen, int(flash_pos[0]), int(flash_pos[1]), 8, self.COLOR_MUZZLE_FLASH)
            pygame.gfxdraw.filled_circle(self.screen, int(flash_pos[0]), int(flash_pos[1]), 5, (255,255,255))

    def _render_zombies(self):
        for z in self.zombies:
            zx, zy = int(z["pos"][0]), int(z["pos"][1])
            # Bobbing animation for walking
            bob = math.sin(z["walk_phase"] + self.steps * 0.5) * 2
            
            body_rect = pygame.Rect(zx - 8, zy - 30 + bob, 16, 22)
            head_pos = (zx, zy - 35 + bob)
            
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, body_rect, border_radius=3)
            pygame.gfxdraw.filled_circle(self.screen, int(head_pos[0]), int(head_pos[1]), 7, self.COLOR_ZOMBIE)

    def _render_bullets(self):
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, b["rect"], border_radius=2)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["lifetime"] / 5))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"HP: {self.player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        health_bar_bg = pygame.Rect(10, 40, 102, 12)
        health_bar_fg = pygame.Rect(11, 41, int(100 * (self.player_health/self.PLAYER_HEALTH_MAX)), 10)
        pygame.draw.rect(self.screen, (0,0,0), health_bar_bg)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, health_bar_fg)

        # Ammo
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 60))
        ammo_bar_bg = pygame.Rect(10, 90, 102, 12)
        ammo_bar_fg = pygame.Rect(11, 91, int(100 * (self.player_ammo/self.PLAYER_AMMO_MAX)), 10)
        pygame.draw.rect(self.screen, (0,0,0), ammo_bar_bg)
        pygame.draw.rect(self.screen, self.COLOR_AMMO, ammo_bar_fg)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = self.font_ui.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU SURVIVED!"
                color = self.COLOR_PLAYER
            else:
                msg = "YOU DIED"
                color = self.COLOR_HEALTH
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1

        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS) # Control game speed for human play
        
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    
    # Show final screen for a moment
    pygame.time.wait(2000)
    
    env.close()