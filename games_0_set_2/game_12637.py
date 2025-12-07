import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:01:32.556054
# Source Brief: brief_02637.md
# Brief Index: 2637
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A cyberpunk-themed stealth platformer where the player manipulates glitching
    cityscapes to create cover and eliminate enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A cyberpunk-themed stealth platformer. Manipulate glitching cityscapes to create cover, "
        "eliminate enemies, and upgrade your weapon."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shoot and shift to glitch platforms for cover."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_BG_ACCENT = (25, 15, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (150, 20, 20)
    COLOR_ENERGY = (255, 255, 0)
    COLOR_COVER = (50, 255, 100)
    COLOR_PLATFORM = (100, 80, 200)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 255, 120)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_SPEED = 6
    PLAYER_HEALTH_MAX = 100
    PLAYER_SHOOT_COOLDOWN = 10  # frames
    PLAYER_GLITCH_COOLDOWN = 30 # frames

    # Enemy settings
    ENEMY_SIZE = 14
    ENEMY_BASE_HEALTH = 50
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SHOOT_COOLDOWN = 45 # frames
    ENEMY_DETECTION_RANGE = 250
    ENEMY_ALERT_DURATION = 150 # frames

    # Game settings
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode
        self.current_level = 0
        self.last_win = False

        self._initialize_state_variables()
        self.reset()

    def _initialize_state_variables(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.player = {}
        self.enemies = []
        self.platforms = []
        self.energy_pickups = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.shoot_cooldown = 0
        self.glitch_cooldown = 0
        
        self.background_elements = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.last_win:
            self.current_level += 1
        else:
            self.current_level = 1
        self.last_win = False

        self._initialize_state_variables()
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # Apply terminal rewards
        if terminated and not truncated:
            if len(self.enemies) == 0: # Victory
                self.reward_this_step += 100
                self.last_win = True
            elif self.player['health'] <= 0: # Defeat
                self.reward_this_step -= 100
        
        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Core Game Logic ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        vel = [0, 0]
        if movement == 1: vel[1] = -self.PLAYER_SPEED  # Up
        elif movement == 2: vel[1] = self.PLAYER_SPEED   # Down
        elif movement == 3: vel[0] = -self.PLAYER_SPEED  # Left
        elif movement == 4: vel[0] = self.PLAYER_SPEED   # Right
        
        self.player['pos'][0] += vel[0]
        self.player['pos'][1] += vel[1]
        
        if vel[0] != 0 or vel[1] != 0:
            self.player['last_move_dir'] = [v / self.PLAYER_SPEED if self.PLAYER_SPEED != 0 else 0 for v in vel]

        # Shooting
        if space_held and self.shoot_cooldown == 0:
            self._player_shoot()
            self.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
        
        # Glitching
        if shift_held and self.glitch_cooldown == 0:
            self._player_glitch()
            self.glitch_cooldown = self.PLAYER_GLITCH_COOLDOWN

    def _update_game_state(self):
        # Cooldowns
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.glitch_cooldown > 0: self.glitch_cooldown -= 1

        # Player
        self._update_player()
        
        # Enemies
        for enemy in self.enemies:
            self._update_enemy(enemy)
        
        # Projectiles
        self._update_projectiles()
        
        # Particles
        self._update_particles()

        # Cleanup
        self.player_projectiles = [p for p in self.player_projectiles if p['alive']]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p['alive']]
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.energy_pickups = [e for e in self.energy_pickups if e['alive']]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_player(self):
        # Boundary checks
        self.player['pos'][0] = np.clip(self.player['pos'][0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player['pos'][1] = np.clip(self.player['pos'][1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Collision with solid platforms
        player_rect = pygame.Rect(self.player['pos'][0] - self.PLAYER_SIZE, self.player['pos'][1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        for plat in self.platforms:
            if plat['glitched']:
                plat_rect = pygame.Rect(plat['rect'])
                if player_rect.colliderect(plat_rect):
                    # Simple push-out collision response
                    # Determine overlap on each axis
                    dx = min(player_rect.right - plat_rect.left, plat_rect.right - player_rect.left)
                    dy = min(player_rect.bottom - plat_rect.top, plat_rect.bottom - player_rect.top)

                    if dx < dy:
                        if player_rect.centerx < plat_rect.centerx:
                            self.player['pos'][0] -= dx
                        else:
                            self.player['pos'][0] += dx
                    else:
                        if player_rect.centery < plat_rect.centery:
                            self.player['pos'][1] -= dy
                        else:
                            self.player['pos'][1] += dy
        
        # Collect energy
        for energy in self.energy_pickups:
            if math.hypot(self.player['pos'][0] - energy['pos'][0], self.player['pos'][1] - energy['pos'][1]) < self.PLAYER_SIZE + 5:
                energy['alive'] = False
                self.player['weapon_level'] += 1
                self.reward_this_step += 0.5
                self._create_particles(energy['pos'], self.COLOR_ENERGY, 20, 3)

    def _update_enemy(self, enemy):
        # State transitions and movement
        player_visible, dist_to_player = self._check_line_of_sight(enemy['pos'], self.player['pos'])

        if enemy['alert_timer'] > 0:
            enemy['alert_timer'] -= 1
        if enemy['alert_timer'] == 0 and enemy['alert_status'] == 'alerted':
            enemy['alert_status'] = 'calm'

        if player_visible and dist_to_player < self.ENEMY_DETECTION_RANGE:
            if enemy['alert_status'] == 'calm':
                self.reward_this_step -= 0.1 # First detection penalty
            enemy['alert_status'] = 'alerted'
            enemy['alert_timer'] = self.ENEMY_ALERT_DURATION
            enemy['last_known_player_pos'] = list(self.player['pos'])
        
        # Movement and shooting
        if enemy['alert_status'] == 'alerted':
            # Face player and shoot
            if enemy['shoot_cooldown'] == 0:
                self._enemy_shoot(enemy)
                enemy['shoot_cooldown'] = self.ENEMY_SHOOT_COOLDOWN + random.randint(-5, 5)
        else: # Calm, patrol
            target_waypoint = enemy['patrol_path'][enemy['waypoint_idx']]
            dist = math.hypot(target_waypoint[0] - enemy['pos'][0], target_waypoint[1] - enemy['pos'][1])
            if dist < enemy['speed'] * 2:
                enemy['waypoint_idx'] = (enemy['waypoint_idx'] + 1) % len(enemy['patrol_path'])
            else:
                angle = math.atan2(target_waypoint[1] - enemy['pos'][1], target_waypoint[0] - enemy['pos'][0])
                enemy['pos'][0] += math.cos(angle) * enemy['speed']
                enemy['pos'][1] += math.sin(angle) * enemy['speed']

        if enemy['shoot_cooldown'] > 0:
            enemy['shoot_cooldown'] -= 1

    def _update_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p_rect = pygame.Rect(p['pos'][0] - 2, p['pos'][1] - 2, 4, 4)

            # Check collision with enemies
            for enemy in self.enemies:
                if math.hypot(p['pos'][0] - enemy['pos'][0], p['pos'][1] - enemy['pos'][1]) < self.ENEMY_SIZE + 2:
                    p['alive'] = False
                    damage = 10 * self.player['weapon_level']
                    enemy['health'] -= damage
                    self.reward_this_step += 0.1
                    self._create_particles(p['pos'], self.COLOR_ENEMY, 10, 2)
                    if enemy['health'] <= 0:
                        self.reward_this_step += 1.0
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 50, 5)
                    break
            
            # Check collision with platforms and bounds
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                p['alive'] = False
            if any(pygame.Rect(plat['rect']).colliderect(p_rect) for plat in self.platforms if plat['glitched']):
                p['alive'] = False
                self._create_particles(p['pos'], self.COLOR_COVER, 5, 1)

        # Enemy projectiles
        for p in self.enemy_projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p_rect = pygame.Rect(p['pos'][0] - 3, p['pos'][1] - 3, 6, 6)

            # Check collision with player
            if math.hypot(p['pos'][0] - self.player['pos'][0], p['pos'][1] - self.player['pos'][1]) < self.PLAYER_SIZE + 3:
                p['alive'] = False
                self.player['health'] -= 10 * (1 + (self.current_level - 1) * 0.05)
                self._create_particles(p['pos'], self.COLOR_PLAYER, 15, 4)
            
            # Check collision with platforms and bounds
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                p['alive'] = False
            if any(pygame.Rect(plat['rect']).colliderect(p_rect) for plat in self.platforms if plat['glitched']):
                p['alive'] = False
                self._create_particles(p['pos'], self.COLOR_COVER, 5, 1)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # gravity
            p['life'] -= 1

    # --- Actions ---

    def _player_shoot(self):
        proj_speed = 10 + self.player['weapon_level']
        vel = [d * proj_speed for d in self.player['last_move_dir']]
        if not any(vel): # Don't shoot if not moving
            return
        self.player_projectiles.append({
            'pos': list(self.player['pos']),
            'vel': vel,
            'alive': True
        })
        # Muzzle flash particles
        self._create_particles(self.player['pos'], self.COLOR_PLAYER, 5, 4, direction=self.player['last_move_dir'])

    def _enemy_shoot(self, enemy):
        target_pos = enemy['last_known_player_pos']
        angle = math.atan2(target_pos[1] - enemy['pos'][1], target_pos[0] - enemy['pos'][0])
        proj_speed = 8
        vel = [math.cos(angle) * proj_speed, math.sin(angle) * proj_speed]
        self.enemy_projectiles.append({
            'pos': list(enemy['pos']),
            'vel': vel,
            'alive': True
        })
    
    def _player_glitch(self):
        if not self.platforms: return

        player_pos = pygame.Vector2(self.player['pos'])
        closest_plat = min(self.platforms, key=lambda p: player_pos.distance_to(pygame.Rect(p['rect']).center))
        
        closest_plat['glitched'] = not closest_plat['glitched']
        self._create_particles(pygame.Rect(closest_plat['rect']).center, self.COLOR_COVER if closest_plat['glitched'] else self.COLOR_PLATFORM, 30, 4)

    # --- State Generation & Checks ---

    def _generate_level(self):
        # Player
        self.player = {
            'pos': [100, self.HEIGHT - 50],
            'health': self.PLAYER_HEALTH_MAX,
            'weapon_level': 1,
            'last_move_dir': [1, 0]
        }

        # Platforms
        self.platforms = [
            {'rect': (50, 300, 150, 20), 'glitched': False},
            {'rect': (250, 220, 150, 20), 'glitched': False},
            {'rect': (450, 150, 150, 20), 'glitched': False},
            {'rect': (250, 350, 150, 20), 'glitched': True}, # Start with one cover
            {'rect': (50, 100, 150, 20), 'glitched': False},
        ]
        
        # Enemies
        num_enemies = min(4, 1 + (self.current_level - 1) // 2)
        enemy_health = self.ENEMY_BASE_HEALTH * (1 + (self.current_level - 1) * 0.05)
        enemy_speed = self.ENEMY_BASE_SPEED + (self.current_level - 1) * 0.05

        patrol_paths = [
            [(500, 100), (600, 100)],
            [(100, 50), (300, 50)],
            [(400, 300), (550, 300)],
            [(100, 200), (200, 200)]
        ]
        random.shuffle(patrol_paths)

        for i in range(num_enemies):
            path = patrol_paths[i]
            self.enemies.append({
                'pos': list(path[0]),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'patrol_path': path,
                'waypoint_idx': 0,
                'alert_status': 'calm', # calm, alerted
                'alert_timer': 0,
                'last_known_player_pos': [0,0],
                'shoot_cooldown': 0,
            })
        
        # Energy Pickups
        self.energy_pickups = [
            {'pos': [325, 200], 'alive': True},
            {'pos': [125, 80], 'alive': True}
        ]

        # Background
        if not self.background_elements:
            for _ in range(50):
                x1, y1 = random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)
                length = random.randint(20, 100)
                angle = random.choice([0, math.pi/2])
                x2 = x1 + math.cos(angle) * length
                y2 = y1 + math.sin(angle) * length
                self.background_elements.append(((x1, y1), (x2, y2), random.randint(1, 3)))

    def _check_termination(self):
        if self.player['health'] <= 0: return True
        if not self.enemies: return True
        return False

    def _check_line_of_sight(self, start_pos, end_pos):
        start_vec = pygame.Vector2(start_pos)
        end_vec = pygame.Vector2(end_pos)
        dist = start_vec.distance_to(end_vec)
        
        for plat in self.platforms:
            if plat['glitched']:
                if pygame.Rect(plat['rect']).clipline(start_pos, end_pos):
                    return False, dist
        return True, dist

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_platforms()
        self._render_energy_pickups()
        self._render_enemies()
        self._render_player()
        self._render_projectiles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for start, end, width in self.background_elements:
            pygame.draw.line(self.screen, self.COLOR_BG_ACCENT, start, end, width)

    def _render_platforms(self):
        for plat in self.platforms:
            rect = pygame.Rect(plat['rect'])
            if plat['glitched']:
                pygame.draw.rect(self.screen, self.COLOR_COVER, rect)
                # Add a subtle glow/edge
                pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_COVER), rect, 1)
            else:
                # Flickering outline for non-glitched platforms
                flicker_color = list(self.COLOR_PLATFORM)
                flicker_color[2] = int(150 + math.sin(self.steps * 0.5 + rect.x) * 50)
                pygame.draw.rect(self.screen, tuple(flicker_color), rect, 2)

    def _render_energy_pickups(self):
        for energy in self.energy_pickups:
            pulse = abs(math.sin(self.steps * 0.1))
            radius = int(5 + pulse * 3)
            glow_radius = int(radius * (1.5 + pulse))
            
            # Glow
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_ENERGY, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (int(energy['pos'][0] - glow_radius), int(energy['pos'][1] - glow_radius)))
            
            # Core
            pygame.draw.circle(self.screen, self.COLOR_ENERGY, (int(energy['pos'][0]), int(energy['pos'][1])), radius)

    def _render_player(self):
        pos = (int(self.player['pos'][0]), int(self.player['pos'][1]))
        self._draw_glowing_circle(pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.PLAYER_SIZE)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            self._draw_glowing_circle(pos, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, self.ENEMY_SIZE)
            
            # Health bar
            bar_w, bar_h = 30, 4
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            bg_rect = (pos[0] - bar_w/2, pos[1] - self.ENEMY_SIZE - 10, bar_w, bar_h)
            fill_rect = (pos[0] - bar_w/2, pos[1] - self.ENEMY_SIZE - 10, bar_w * health_pct, bar_h)
            pygame.draw.rect(self.screen, (50,50,50), bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect)

            # Alert status
            if enemy['alert_status'] == 'alerted':
                alert_color = self.COLOR_ENEMY
            else:
                alert_color = self.COLOR_COVER
            
            alert_y = pos[1] - self.ENEMY_SIZE - 15
            pygame.draw.circle(self.screen, alert_color, (pos[0], int(alert_y)), 3)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            self._draw_glowing_circle(pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 2)
        for p in self.enemy_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            self._draw_glowing_circle(pos, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['life'] / p['max_life'])
            color = (*p['color'], int(alpha * 255))
            size = int(p['size'] * alpha)
            if size > 0 and pygame.get_init():
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)
                except (TypeError, ValueError): # Can happen if screen is closed
                    pass


    def _render_ui(self):
        # Health Bar
        bar_w, bar_h = 200, 20
        health_pct = max(0, self.player['health'] / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, self.HEIGHT - 30, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, self.HEIGHT - 30, bar_w * health_pct, bar_h))
        self._draw_text(f"HP: {int(self.player['health'])}", (15, self.HEIGHT - 29), self.font_small, (0,0,0))
        
        # Weapon Level
        self._draw_text(f"WEAPON LVL: {self.player['weapon_level']}", (10, 10), self.font_large, self.COLOR_UI_TEXT)
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

    # --- Helper Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "player_health": self.player.get('health', 0),
            "enemies_left": len(self.enemies),
            "weapon_level": self.player.get('weapon_level', 0),
        }

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_glowing_circle(self, pos, color, glow_color, radius):
        glow_radius = int(radius * 1.8)
        
        # Use a surface for alpha blending the glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Main circle
        pygame.draw.circle(self.screen, color, pos, radius)

    def _create_particles(self, pos, color, count, max_speed, direction=None):
        for _ in range(count):
            if direction:
                angle = math.atan2(direction[1], direction[0]) + random.uniform(-0.5, 0.5)
            else:
                angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(1, 4)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Set the video driver to a real one for manual play
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'macOS' etc.
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Cyberglitch Stealth")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping for human play
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
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}, Level: {info['level']}")
            # Reset and continue playing for demonstration
            obs, info = env.reset()
            done = False
            total_reward = 0
            
    env.close()