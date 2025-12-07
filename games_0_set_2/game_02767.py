import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Hold Space to shoot. Survive the horde!"
    )

    game_description = (
        "A fast-paced, side-scrolling zombie survival shooter. Mow down hordes of the undead, "
        "collect ammo, and survive for as long as you can as the difficulty ramps up."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = self.WIDTH * 2
        self.GROUND_Y = 350
        self.FPS = 30

        # Colors
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_CITY_BG = (42, 44, 60)
        self.COLOR_GROUND = (51, 51, 51)
        self.COLOR_PLAYER = (255, 140, 0)
        self.COLOR_ZOMBIE = (90, 110, 90)
        self.COLOR_ZOMBIE_HEAD = (70, 90, 70)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_AMMO = (0, 255, 0)
        self.COLOR_HEALTH = (255, 0, 0)
        self.COLOR_UI_BG = (10, 10, 10, 180)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_BLOOD = (139, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 223, 0)

        # Physics & Game Rules
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -15
        self.GRAVITY = 0.8
        self.ZOMBIE_SPEED = 0.75
        self.BULLET_SPEED = 15
        self.SHOOT_COOLDOWN_FRAMES = 5
        self.PLAYER_INVULNERABILITY_FRAMES = 30
        self.MAX_STEPS = 3000

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_timer = pygame.font.Font(None, 32)
        
        # --- State variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.player_health = 0
        self.player_ammo = 0
        self.is_jumping = False
        self.is_ducking = False
        self.facing_right = True
        self.shoot_cooldown = 0
        self.player_invulnerability = 0
        
        self.zombies = []
        self.bullets = []
        self.pickups = []
        self.particles = []
        
        self.camera_x = 0
        self.zombie_spawn_timer = 0
        self.pickup_spawn_timer = 0
        self.zombie_spawn_rate = 0.01

        self.city_buildings = []

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = [100, self.GROUND_Y]
        self.player_vel = [0, 0]
        self.player_health = 100
        self.player_ammo = 50
        self.is_jumping = False
        self.is_ducking = False
        self.facing_right = True
        self.shoot_cooldown = 0
        self.player_invulnerability = 0

        self.zombies = []
        self.bullets = []
        self.pickups = []
        self.particles = []

        self.camera_x = 0
        self.zombie_spawn_timer = 0
        self.pickup_spawn_timer = self.np_random.integers(150, 300)
        self.zombie_spawn_rate = 0.01 # 1 zombie per 100 steps on average

        self._generate_cityscape()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        self._handle_input(action)
        self._update_player()
        
        kill_reward = self._update_bullets()
        reward += kill_reward
        self.score += kill_reward
        
        self._update_zombies()
        
        pickup_reward = self._update_pickups()
        reward += pickup_reward
        self.score += pickup_reward * 2 # score is 1 per pickup

        self._update_particles()
        
        damage_taken = self._handle_collisions()
        if damage_taken > 0 and self.player_invulnerability == 0:
            self.player_health -= damage_taken
            self.player_invulnerability = self.PLAYER_INVULNERABILITY_FRAMES
            # Sound: Player hurt
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2, 20, 360)


        self._spawn_entities()
        self._update_timers()
        self._update_camera()

        self.steps += 1
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100
            else: # Survived
                reward = 100

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        self.player_vel[0] = 0
        if movement == 3: # Left
            self.player_vel[0] = -self.PLAYER_SPEED
            self.facing_right = False
        elif movement == 4: # Right
            self.player_vel[0] = self.PLAYER_SPEED
            self.facing_right = True

        # Vertical Movement
        if movement == 1 and not self.is_jumping: # Jump
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_jumping = True
            # Sound: Jump
            self._create_particles([self.player_pos[0], self.GROUND_Y + 20], 5, (100,100,100), 1, 10, 45, -90-22)


        self.is_ducking = (movement == 2)

        # Shooting
        if space_held and self.shoot_cooldown == 0 and self.player_ammo > 0:
            self.player_ammo -= 1
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES
            bullet_dir = 1 if self.facing_right else -1
            gun_pos_y = self.player_pos[1] - 25 - (10 if self.is_ducking else 0)
            gun_pos_x = self.player_pos[0] + (20 * bullet_dir)
            self.bullets.append({'pos': [gun_pos_x, gun_pos_y], 'dir': bullet_dir})
            # Sound: Shoot
            # Muzzle flash
            self._create_particles([gun_pos_x, gun_pos_y], 1, self.COLOR_MUZZLE_FLASH, 5, 3, 0)


    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # World bounds collision
        self.player_pos[0] = max(10, min(self.player_pos[0], self.WORLD_WIDTH - 10))

        # Ground collision
        player_height = 20 if self.is_ducking else 40
        if self.player_pos[1] > self.GROUND_Y - player_height / 2:
            self.player_pos[1] = self.GROUND_Y - player_height / 2
            self.player_vel[1] = 0
            if self.is_jumping:
                self.is_jumping = False
                # Sound: Land
                self._create_particles([self.player_pos[0], self.GROUND_Y + 20], 8, (100,100,100), 1, 10, 45, -90-22)


    def _update_bullets(self):
        reward = 0
        for b in self.bullets[:]:
            b['pos'][0] += self.BULLET_SPEED * b['dir']
            if not (0 < b['pos'][0] < self.WORLD_WIDTH):
                self.bullets.remove(b)
                continue
            
            for z in self.zombies[:]:
                z_pos, z_size = z['pos'], z['size']
                if (z_pos[0] - z_size[0]/2 < b['pos'][0] < z_pos[0] + z_size[0]/2 and
                    z_pos[1] - z_size[1] < b['pos'][1] < z_pos[1]):
                    z['health'] -= 10
                    self._create_particles(b['pos'], 10, self.COLOR_BLOOD, 2, 15, 360)
                    if b in self.bullets: self.bullets.remove(b)
                    if z['health'] <= 0:
                        reward += 1
                        self.zombies.remove(z)
                        # Sound: Zombie die
                        self._create_particles(z['pos'], 30, self.COLOR_BLOOD, 3, 25, 360)
                    else:
                        # Sound: Zombie hit
                        pass
                    break
        return reward

    def _update_zombies(self):
        for z in self.zombies[:]:
            # Despawn if off-screen left
            if z['pos'][0] < -50:
                self.zombies.remove(z)
                continue

            # Move towards player
            dx = self.player_pos[0] - z['pos'][0]
            dist = abs(dx) # Only move horizontally
            if dist > 20: # Stop if close
                z['pos'][0] += (dx / (dist + 1e-6)) * self.ZOMBIE_SPEED
            z['anim_timer'] = (z['anim_timer'] + 1) % 40

    def _update_pickups(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 40, 20, 40)
        for p in self.pickups[:]:
            pickup_rect = pygame.Rect(p['pos'][0] - 10, p['pos'][1] - 10, 20, 20)
            if player_rect.colliderect(pickup_rect):
                self.player_ammo = min(999, self.player_ammo + 20)
                reward += 0.5
                self.pickups.remove(p)
                # Sound: Ammo pickup
        return reward

    def _handle_collisions(self):
        damage = 0
        player_height = 20 if self.is_ducking else 40
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - player_height, 20, player_height)
        
        for z in self.zombies:
            z_rect = pygame.Rect(z['pos'][0] - z['size'][0]/2, z['pos'][1] - z['size'][1], z['size'][0], z['size'][1])
            if player_rect.colliderect(z_rect):
                damage += 10
                # Apply knockback to player
                knockback_dir = 1 if self.player_pos[0] < z['pos'][0] else -1
                self.player_vel[0] = -knockback_dir * 5
                self.player_vel[1] = -3
                break # Only take damage from one zombie per frame
        return damage

    def _spawn_entities(self):
        # Spawn Zombies
        if self.np_random.random() < self.zombie_spawn_rate:
            side = self.np_random.choice([-1, 1])
            spawn_x = self.camera_x - 50 if side == -1 else self.camera_x + self.WIDTH + 50
            spawn_x = max(0, min(self.WORLD_WIDTH, spawn_x))
            
            new_zombie = {
                'pos': [spawn_x, self.GROUND_Y],
                'health': 10,
                'size': (self.np_random.integers(18, 25), self.np_random.integers(35, 45)),
                'anim_timer': 0
            }
            self.zombies.append(new_zombie)
            
        # Spawn Ammo Pickups
        self.pickup_spawn_timer -= 1
        if self.pickup_spawn_timer <= 0:
            spawn_x = self.np_random.integers(int(self.camera_x), int(self.camera_x + self.WIDTH))
            spawn_x = max(0, min(self.WORLD_WIDTH, spawn_x))
            self.pickups.append({'pos': [spawn_x, self.GROUND_Y - 10]})
            self.pickup_spawn_timer = self.np_random.integers(300, 600)

    def _update_timers(self):
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.player_invulnerability = max(0, self.player_invulnerability - 1)
        if self.steps > 0 and self.steps % 100 == 0:
            self.zombie_spawn_rate += 0.001

    def _update_camera(self):
        target_camera_x = self.player_pos[0] - self.WIDTH / 2
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        # Clamp camera to world bounds
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.WIDTH))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax cityscape
        for b in self.city_buildings:
            bx = (b['x'] - self.camera_x * 0.5)
            # Seamlessly loop the background
            if bx + b['w'] < 0:
                b['x'] += self.WORLD_WIDTH
            if bx > self.WIDTH:
                b['x'] -= self.WORLD_WIDTH

            pygame.draw.rect(self.screen, self.COLOR_CITY_BG, (bx, b['y'], b['w'], b['h']))
        
        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

    def _render_game(self):
        # Pickups
        for p in self.pickups:
            screen_x = int(p['pos'][0] - self.camera_x)
            screen_y = int(p['pos'][1])
            pulse = abs(math.sin(self.steps * 0.1))
            radius = int(7 + pulse * 3)
            pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, radius, self.COLOR_AMMO)
            pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, radius, self.COLOR_AMMO)
        
        # Zombies
        for z in self.zombies:
            screen_x = int(z['pos'][0] - self.camera_x)
            w, h = z['size']
            
            # Simple walk animation
            anim_offset = math.sin(z['anim_timer'] * 0.2) * 2
            
            body_rect = pygame.Rect(screen_x - w/2, z['pos'][1] - h + anim_offset, w, h)
            head_rect = pygame.Rect(screen_x - 8, z['pos'][1] - h - 8 + anim_offset, 16, 16)
            
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, body_rect)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEAD, head_rect)

        # Player
        player_height = 20 if self.is_ducking else 40
        player_width = 18
        screen_x = int(self.player_pos[0] - self.camera_x)
        screen_y = int(self.player_pos[1])

        # Bobbing animation for movement
        bob = 0
        if not self.is_jumping and self.player_vel[0] != 0:
            bob = math.sin(self.steps * 0.5) * 2
        
        player_rect = pygame.Rect(
            screen_x - player_width/2, 
            screen_y - player_height + bob, 
            player_width, 
            player_height
        )

        # Flash if invulnerable
        if self.player_invulnerability > 0 and self.steps % 4 < 2:
            pass # Don't draw player
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

        # Bullets
        for b in self.bullets:
            screen_x = int(b['pos'][0] - self.camera_x)
            pygame.draw.rect(self.screen, self.COLOR_BULLET, (screen_x-3, b['pos'][1]-1, 6, 2))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            screen_x = int(p['pos'][0] - self.camera_x)
            screen_y = int(p['pos'][1])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            
            size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
            
            # Use a surface for alpha blending
            part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(part_surf, color, (size, size), size)
            self.screen.blit(part_surf, (screen_x - size, screen_y - size))
            

    def _render_ui(self):
        # UI Background panels
        s = pygame.Surface((160, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (10, 10)) # Health
        self.screen.blit(s, (self.WIDTH - 170, 10)) # Ammo

        # Health Bar
        health_ratio = max(0, self.player_health / 100)
        health_bar_w = int(140 * health_ratio)
        pygame.draw.rect(self.screen, (80,0,0), (20, 25, 140, 15))
        if health_bar_w > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (20, 25, health_bar_w, 15))
        health_text = self.font_ui.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (20, 12))

        # Ammo Count
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH - 160, 20))

        # Timer / Steps
        time_text = self.font_timer.render(f"SURVIVAL TIME: {self.steps}", True, self.COLOR_TEXT)
        text_rect = time_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 25))
        self.screen.blit(time_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }

    def _generate_cityscape(self):
        self.city_buildings = []
        current_x = -self.WIDTH / 2
        while current_x < self.WORLD_WIDTH + self.WIDTH:
            w = self.np_random.integers(50, 150)
            h = self.np_random.integers(50, 250)
            self.city_buildings.append({'x': current_x, 'y': self.GROUND_Y - h, 'w': w, 'h': h})
            current_x += w + self.np_random.integers(10, 50)

    def _create_particles(self, pos, num, color, size, life, spread_angle=360, base_angle=0):
        for _ in range(num):
            angle = math.radians(self.np_random.uniform(base_angle - spread_angle/2, base_angle + spread_angle/2))
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'size': size,
                'life': life,
                'max_life': life
            })

    def validate_implementation(self):
        print("Running implementation validation...")
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
        
        # Test state assertions
        assert self.player_health <= 100
        assert self.player_ammo >= 0
        assert self.steps >= 0

        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()