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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. ↑↓ to aim. Press space to shoot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in a side-scrolling shooter. Collect ammo and hold out for as long as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_GROUND = (45, 40, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_ZOMBIE = (80, 110, 80)
    COLOR_BULLET = (255, 255, 0)
    COLOR_AMMO_CRATE = (0, 255, 100)
    COLOR_ENEMY_PROJECTILE = (255, 100, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (255, 0, 0)

    # Game parameters
    GROUND_Y = 350
    PLAYER_SPEED = 5
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 50
    PLAYER_FIRE_COOLDOWN = 5 # steps

    BULLET_SPEED = 15
    
    ZOMBIE_SPEED = 1
    ZOMBIE_WIDTH = 20
    ZOMBIE_HEIGHT = 40
    ZOMBIE_HEALTH = 30
    ZOMBIE_ATTACK_COOLDOWN = 60 # steps
    ZOMBIE_PROJECTILE_SPEED = 4

    AMMO_CRATE_VALUE = 25
    AMMO_SPAWN_INTERVAL = 400 # steps

    MAX_STEPS = 3000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.np_random = None # Will be set in reset
        self.player = {}
        self.zombies = []
        self.bullets = []
        self.enemy_projectiles = []
        self.ammo_crates = []
        self.particles = []
        
        self.steps = 0
        self.episode_reward = 0
        self.game_over = False
        self.zombie_kill_count = 0
        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = 0

        # self.reset() is not called here to avoid initializing np_random before a seed is passed
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = {
            'rect': pygame.Rect(self.SCREEN_WIDTH // 2, self.GROUND_Y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT),
            'health': self.PLAYER_MAX_HEALTH,
            'ammo': self.PLAYER_MAX_AMMO,
            'aim_angle': 0,
            'facing': 1, # 1 for right, -1 for left
            'last_shot': -self.PLAYER_FIRE_COOLDOWN
        }

        self.zombies = []
        self.bullets = []
        self.enemy_projectiles = []
        self.ammo_crates = []
        self.particles = []

        self.steps = 0
        self.episode_reward = 0
        self.game_over = False
        self.zombie_kill_count = 0
        self.zombie_spawn_timer = 100
        self.ammo_spawn_timer = self.AMMO_SPAWN_INTERVAL
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement and Aiming
        if movement == 1: # Aim up
            self.player['aim_angle'] = min(math.pi / 4, self.player['aim_angle'] + 0.1)
        elif movement == 2: # Aim down
            self.player['aim_angle'] = max(-math.pi / 4, self.player['aim_angle'] - 0.1)
        elif movement == 3: # Left
            self.player['rect'].x -= self.PLAYER_SPEED
            self.player['facing'] = -1
        elif movement == 4: # Right
            self.player['rect'].x += self.PLAYER_SPEED
            self.player['facing'] = 1
        
        # Clamp player position
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.SCREEN_WIDTH, self.player['rect'].right)

        # Shooting
        if space_held and self.player['ammo'] > 0 and (self.steps - self.player['last_shot']) >= self.PLAYER_FIRE_COOLDOWN:
            self.player['last_shot'] = self.steps
            self.player['ammo'] -= 1
            reward -= 0.01 # Penalty for shooting, encouraging accuracy

            angle = self.player['aim_angle']
            if self.player['facing'] == -1:
                angle = math.pi - angle

            start_pos = self.player['rect'].center
            vel = (math.cos(angle) * self.BULLET_SPEED, math.sin(angle) * self.BULLET_SPEED)
            self.bullets.append({'pos': list(start_pos), 'vel': vel})
            # Sound effect placeholder: # pew!
            self._create_muzzle_flash(start_pos, angle)

        # --- Update Game State ---
        reward += self._update_bullets()
        reward += self._update_zombies()
        reward += self._update_enemy_projectiles()
        reward += self._update_ammo_crates()
        self._update_particles()
        
        # --- Spawning ---
        self._spawn_zombies()
        self._spawn_ammo()

        # --- Check Termination ---
        terminated = self.player['health'] <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player['health'] > 0 and self.steps >= self.MAX_STEPS:
                reward += 100 # Survival bonus
        
        self.episode_reward += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_bullets(self):
        reward = 0
        for b in self.bullets[:]:
            b['pos'][0] += b['vel'][0]
            b['pos'][1] += b['vel'][1]

            if not (0 < b['pos'][0] < self.SCREEN_WIDTH and 0 < b['pos'][1] < self.SCREEN_HEIGHT):
                self.bullets.remove(b)
                continue
            
            bullet_rect = pygame.Rect(b['pos'][0]-2, b['pos'][1]-2, 4, 4)
            for z in self.zombies:
                if z['rect'].colliderect(bullet_rect):
                    z['health'] -= 10
                    reward += 0.1 # Hit reward
                    self._create_particles(b['pos'], self.COLOR_ENEMY_PROJECTILE, 5, 2) # Blood splatter
                    if b in self.bullets:
                        self.bullets.remove(b)
                    break
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies[:]:
            # Movement
            if z['rect'].centerx < self.player['rect'].centerx:
                z['rect'].x += self.ZOMBIE_SPEED
            else:
                z['rect'].x -= self.ZOMBIE_SPEED
            
            # Check for death
            if z['health'] <= 0:
                reward += 1 # Kill reward
                self.zombie_kill_count += 1
                self._create_particles(z['rect'].center, self.COLOR_ZOMBIE, 20, 4) # Death explosion
                self.zombies.remove(z)
                # Sound effect placeholder: # squish!
                continue
            
            # Attack
            if (self.steps - z['last_attack']) >= self.ZOMBIE_ATTACK_COOLDOWN:
                z['last_attack'] = self.steps
                dx = self.player['rect'].centerx - z['rect'].centerx
                dy = self.player['rect'].centery - z['rect'].centery
                dist = math.hypot(dx, dy)
                if dist > 0:
                    speed_multiplier = self.ZOMBIE_PROJECTILE_SPEED + (self.steps // 500) * 0.01
                    vel = (dx / dist * speed_multiplier, dy / dist * speed_multiplier)
                    self.enemy_projectiles.append({'pos': list(z['rect'].center), 'vel': vel})
        return reward

    def _update_enemy_projectiles(self):
        for p in self.enemy_projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                self.enemy_projectiles.remove(p)
                continue
            
            proj_rect = pygame.Rect(p['pos'][0]-3, p['pos'][1]-3, 6, 6)
            if self.player['rect'].colliderect(proj_rect):
                self.player['health'] -= 10
                self._create_particles(self.player['rect'].center, self.COLOR_HEALTH_BAR_FG, 10, 3)
                self.enemy_projectiles.remove(p)
                # Sound effect placeholder: # player hit!
        return 0

    def _update_ammo_crates(self):
        reward = 0
        for crate in self.ammo_crates[:]:
            if self.player['rect'].colliderect(crate):
                self.player['ammo'] = min(self.PLAYER_MAX_AMMO, self.player['ammo'] + self.AMMO_CRATE_VALUE)
                reward += 0.5
                self.ammo_crates.remove(crate)
                # Sound effect placeholder: # ammo get!
        return reward

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            spawn_rate_modifier = 1 + self.steps * 0.001
            self.zombie_spawn_timer = max(10, 100 / spawn_rate_modifier)
            
            side = self.np_random.choice([-1, 1])
            x = -self.ZOMBIE_WIDTH if side == -1 else self.SCREEN_WIDTH
            y = self.GROUND_Y - self.ZOMBIE_HEIGHT
            
            new_zombie = {
                'rect': pygame.Rect(x, y, self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT),
                'health': self.ZOMBIE_HEALTH,
                'last_attack': self.steps + self.np_random.integers(0, self.ZOMBIE_ATTACK_COOLDOWN)
            }
            self.zombies.append(new_zombie)

    def _spawn_ammo(self):
        self.ammo_spawn_timer -= 1
        if self.ammo_spawn_timer <= 0:
            self.ammo_spawn_timer = self.AMMO_SPAWN_INTERVAL
            if len(self.ammo_crates) < 2:
                x = self.np_random.integers(50, self.SCREEN_WIDTH - 50)
                y = self.GROUND_Y - 15
                self.ammo_crates.append(pygame.Rect(x, y, 15, 15))

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
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        # Background cityscape
        for i in range(3):
            color = tuple(c * (0.6 - i * 0.1) for c in self.COLOR_GROUND)
            offset = (self.steps * (i+1) * 0.05) % self.SCREEN_WIDTH
            for j in range(self.np_random.integers(5,10)):
                w = self.np_random.integers(20, 80)
                h = self.np_random.integers(50, 200 - i * 40)
                x = (j * 150 + offset) % (self.SCREEN_WIDTH + w) - w
                pygame.draw.rect(self.screen, color, (x, self.GROUND_Y - h, w, h))

        # Ammo Crates
        for crate in self.ammo_crates:
            pygame.draw.rect(self.screen, self.COLOR_AMMO_CRATE, crate)
            pygame.draw.rect(self.screen, (255,255,255), crate.inflate(2,2), 1)

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z['rect'])

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player['rect'])
        
        # Aiming line
        angle = self.player['aim_angle']
        if self.player['facing'] == -1: angle = math.pi - angle
        start_pos = self.player['rect'].center
        end_pos = (start_pos[0] + 30 * math.cos(angle), start_pos[1] + 30 * math.sin(angle))
        pygame.draw.line(self.screen, self.COLOR_PLAYER, start_pos, end_pos, 2)

        # Bullets
        for b in self.bullets:
            pos = (int(b['pos'][0]), int(b['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_BULLET)

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJECTILE)
        
        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size_int = int(p['size'])
            if size_int <= 0:
                continue

            alpha_color = p['color'] + (int(p['alpha']),)
            # The surface size must be an integer tuple. size_int * 2 ensures this.
            surface_size = size_int * 2
            temp_surf = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            # The circle's center and radius must be integers.
            pygame.gfxdraw.filled_circle(temp_surf, size_int, size_int, size_int, alpha_color)
            # The blit position must be integers for correct alignment.
            self.screen.blit(temp_surf, (pos[0] - size_int, pos[1] - size_int))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player['health'] / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 150, 20))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(150 * health_ratio), 20))

        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player['ammo']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (170, 12))

        # Time Survived
        time_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Zombie Kill Count
        kill_text = self.font_small.render(f"KILLS: {self.zombie_kill_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(kill_text, (self.SCREEN_WIDTH - kill_text.get_width() - 10, self.SCREEN_HEIGHT - kill_text.get_height() - 10))

        # Game Over Message
        if self.game_over:
            msg = "SURVIVED" if self.player['health'] > 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.episode_reward,
            "steps": self.steps,
            "kills": self.zombie_kill_count,
            "player_health": self.player['health'],
            "player_ammo": self.player['ammo'],
        }

    # --- Particle System ---
    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'size': self.np_random.integers(3, 7),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _create_muzzle_flash(self, pos, angle):
        for i in range(5):
            speed = self.np_random.uniform(2, 6)
            particle_angle = angle + self.np_random.uniform(-0.5, 0.5)
            vel = [math.cos(particle_angle) * speed, math.sin(particle_angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'size': self.np_random.integers(4, 8),
                'life': 5,
                'color': (255, 220, 100)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.9 # friction
            p['vel'][1] *= 0.9
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.5)
            p['alpha'] = max(0, 255 * (p['life'] / 20))
            if p['life'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    # To run this, you may need to comment out the `os.environ` line at the top.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while not done:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Score: {total_reward:.2f}, Kills: {info['kills']}, Steps: {info['steps']}")
    
    # Keep window open for a few seconds after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)

    env.close()