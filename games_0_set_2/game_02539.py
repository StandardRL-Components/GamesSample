
# Generated: 2025-08-28T05:12:05.074232
# Source Brief: brief_02539.md
# Brief Index: 2539

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire. Press Shift to switch weapons."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 5 minutes against an ever-growing horde of zombies in a top-down arena shooter."
    )

    # auto_advance is True for real-time games with smooth graphics
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5 * 60 * 10  # 5 minutes at 10 steps/sec = 3000 steps
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 3.5
        self.ZOMBIE_SIZE = 18
        self.MAX_HEALTH = 100
        self.MAX_SHOTGUN_AMMO = 100

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_ARENA = (40, 40, 55)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_PISTOL_PROJ = (255, 255, 0)
        self.COLOR_SHOTGUN_PROJ = (255, 180, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_FG = (0, 200, 80)
        self.COLOR_HEALTH_BG = (120, 0, 0)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_weapon = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.shotgun_ammo = 0
        self.active_weapon = 0  # 0: Pistol, 1: Shotgun
        self.last_move_direction = pygame.Vector2(0, -1)
        self.weapon_cooldown = 0
        self.was_shift_held = False

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = 0
        self.zombie_base_speed = 0
        self.difficulty_tier = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.MAX_HEALTH
        self.shotgun_ammo = self.MAX_SHOTGUN_AMMO
        self.active_weapon = 0
        self.last_move_direction = pygame.Vector2(0, -1)
        self.weapon_cooldown = 0
        self.was_shift_held = False

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Entity lists
        self.zombies = []
        self.projectiles = []
        self.particles = []

        # Difficulty and spawning
        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = 30  # 1 zombie every 3 seconds (at 10 steps/sec)
        self.zombie_base_speed = 1.0
        self.difficulty_tier = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01  # Small reward for surviving a step

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_projectiles()
            self._update_zombies()
            self._update_spawns()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            if self.weapon_cooldown > 0:
                self.weapon_cooldown -= 1
        
        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100
                self._create_particles(self.player_pos, (255, 215, 0), 100, 5, 60, is_circle=True)
            else:
                reward -= 100
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 100, 5, 60, is_circle=True)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement and Aiming
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.last_move_direction = move_vec.copy()

        # Weapon Switching (edge-triggered)
        if shift_held and not self.was_shift_held:
            self.active_weapon = 1 - self.active_weapon
            # sound: weapon_switch.wav
        self.was_shift_held = shift_held

        # Shooting
        if space_held and self.weapon_cooldown == 0:
            if self.active_weapon == 0: # Pistol
                self.weapon_cooldown = 8
                proj_vel = self.last_move_direction * 10
                self.projectiles.append({
                    'pos': self.player_pos.copy(),
                    'vel': proj_vel,
                    'color': self.COLOR_PISTOL_PROJ,
                    'size': 4,
                    'damage': 1
                })
                # sound: pistol_shot.wav
                self._create_particles(self.player_pos + self.last_move_direction * 15, (255,255,200), 3, 4, 3)

            elif self.active_weapon == 1 and self.shotgun_ammo > 0: # Shotgun
                self.weapon_cooldown = 20
                self.shotgun_ammo -= 1
                for _ in range(6): # 6 pellets
                    angle = math.radians(random.uniform(-15, 15))
                    proj_vel = self.last_move_direction.rotate(math.degrees(angle)) * random.uniform(8, 12)
                    self.projectiles.append({
                        'pos': self.player_pos.copy(),
                        'vel': proj_vel,
                        'color': self.COLOR_SHOTGUN_PROJ,
                        'size': 5,
                        'damage': 5
                    })
                # sound: shotgun_blast.wav
                self._create_particles(self.player_pos + self.last_move_direction * 15, (255,200,100), 8, 6, 5)

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies(self):
        for z in self.zombies:
            direction = (self.player_pos - z['pos']).normalize()
            z['pos'] += direction * self.zombie_base_speed

    def _update_spawns(self):
        # Difficulty scaling
        if self.steps // 600 > self.difficulty_tier:
            self.difficulty_tier = self.steps // 600
            self.zombie_base_speed = min(self.zombie_base_speed + 0.1, self.PLAYER_SPEED - 0.5)
            self.zombie_spawn_interval = max(self.zombie_spawn_interval - 3, 5)

        # Spawning
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.zombie_spawn_interval:
            self.zombie_spawn_timer = 0
            side = random.randint(0, 3)
            if side == 0: pos = pygame.Vector2(random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE)
            elif side == 1: pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE)
            elif side == 2: pos = pygame.Vector2(-self.ZOMBIE_SIZE, random.uniform(0, self.HEIGHT))
            else: pos = pygame.Vector2(self.WIDTH + self.ZOMBIE_SIZE, random.uniform(0, self.HEIGHT))
            
            self.zombies.append({'pos': pos, 'health': 10})

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Zombie-player collision
        for z in self.zombies[:]:
            z_rect = pygame.Rect(z['pos'].x - self.ZOMBIE_SIZE/2, z['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(z_rect):
                self.player_health -= 10
                self.zombies.remove(z)
                # sound: player_hurt.wav
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 15, 3, 15)
                if self.player_health <= 0:
                    self.player_health = 0

        # Projectile-zombie collision
        for p in self.projectiles[:]:
            p_rect = pygame.Rect(p['pos'].x - p['size']/2, p['pos'].y - p['size']/2, p['size'], p['size'])
            for z in self.zombies[:]:
                z_rect = pygame.Rect(z['pos'].x - self.ZOMBIE_SIZE/2, z['pos'].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if z_rect.colliderect(p_rect):
                    z['health'] -= p['damage']
                    # sound: zombie_hit.wav
                    self._create_particles(p['pos'], self.COLOR_ZOMBIE, 5, 2, 10)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if z['health'] <= 0:
                        reward += 1
                        self.score += 1
                        self.zombies.remove(z)
                        # sound: zombie_die.wav
                        self._create_particles(z['pos'], self.COLOR_ZOMBIE, 20, 3, 20)
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed, lifetime, is_circle=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 1) * speed
            self.particles.append({
                'pos': pos.copy() + vel * 2,
                'vel': vel,
                'color': color,
                'radius': random.uniform(2, 5),
                'lifetime': random.uniform(0.5, 1) * lifetime,
                'is_circle': is_circle
            })

    def _check_termination(self):
        if self.player_health <= 0:
            self.win = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.win = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena boundary
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (0, 0, self.WIDTH, self.HEIGHT), 5)

        # Particles
        for p in self.particles:
            color = p['color']
            if len(color) == 4: # Handle alpha
                s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))
            else:
                if p['is_circle']:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])
                else:
                    pygame.draw.rect(self.screen, p['color'], (p['pos'].x - p['radius']/2, p['pos'].y - p['radius']/2, p['radius'], p['radius']))

        # Zombies
        for z in self.zombies:
            pos_x, pos_y = int(z['pos'].x), int(z['pos'].y)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.ZOMBIE_SIZE // 2, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.ZOMBIE_SIZE // 2, self.COLOR_ZOMBIE)

        # Player
        player_x, player_y = int(self.player_pos.x), int(self.player_pos.y)
        glow_radius = int(self.PLAYER_SIZE * 1.2)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_x - glow_radius, player_y - glow_radius))

        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (player_x, player_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Projectiles
        for p in self.projectiles:
            size = int(p['size'])
            pygame.draw.rect(self.screen, p['color'], (p['pos'].x - size/2, p['pos'].y - size/2, size, size))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_ui.render(f"HP: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 10))

        # Weapon/Ammo Info
        if self.active_weapon == 0:
            weapon_name = "Pistol"
            ammo_count = "∞"
        else:
            weapon_name = "Shotgun"
            ammo_count = str(self.shotgun_ammo)
        
        weapon_text = self.font_weapon.render(f"{weapon_name}: {ammo_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(weapon_text, (self.WIDTH - weapon_text.get_width() - 10, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        minutes = time_left // (60 * 10)
        seconds = (time_left % (60 * 10)) // 10
        timer_str = f"{minutes:02}:{seconds:02}"
        timer_text = self.font_timer.render(timer_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, 5))

        # Kills
        kill_text = self.font_ui.render(f"Kills: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(kill_text, (self.WIDTH - kill_text.get_width() - 10, 35))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU SURVIVED!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_gameover.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "shotgun_ammo": self.shotgun_ammo,
            "active_weapon": self.active_weapon,
            "zombie_count": len(self.zombies),
        }

    def close(self):
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or closing
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS

    env.close()