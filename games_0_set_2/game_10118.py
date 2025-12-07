import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:56:53.382040
# Source Brief: brief_00118.md
# Brief Index: 118
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro-futuristic arcade space shooter.
    The player pilots a ship, battles pirates, and collects upgrades.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship in a retro-futuristic arcade shooter. Battle waves of pirates, "
        "collect powerful upgrades, and aim for a high score."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move your ship. Press space to fire your current weapon. "
        "Press shift to cycle through unlocked weapons."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_PIRATE_SCOUT = (255, 50, 50)
    COLOR_PIRATE_FIGHTER = (255, 150, 0)
    COLOR_PIRATE_BRUISER = (255, 220, 0)
    COLOR_UPGRADE = (255, 215, 0)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_BG = (30, 30, 60)
    COLOR_HEALTH_BAR = (0, 255, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)

    # Game parameters
    MAX_STEPS = 5000
    WIN_SCORE = 1000
    PLAYER_MAX_HEALTH = 100
    PLAYER_BASE_SPEED = 6
    PLAYER_FIRE_COOLDOWN = 10  # in steps
    PIRATE_SPAWN_INTERVAL = 60 # in steps

    WEAPONS = {
        "BLASTER": {
            "name": "BLASTER",
            "color": (0, 255, 150),
            "speed": 12,
            "damage": 10,
            "cooldown_mod": 1.0
        },
        "PLASMA": {
            "name": "PLASMA",
            "color": (200, 50, 255),
            "speed": 8,
            "damage": 25,
            "cooldown_mod": 1.5
        },
        "BEAM": {
            "name": "BEAM",
            "color": (0, 220, 255),
            "speed": 20,
            "damage": 7,
            "cooldown_mod": 0.5
        }
    }


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_weapon = pygame.font.SysFont("monospace", 16, bold=True)

        # Initialize state variables to prevent AttributeError
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.player_speed = 0
        self.player_fire_cooldown = 0
        self.unlocked_weapons = []
        self.current_weapon_idx = 0
        self.last_shift_state = 0
        
        self.pirates = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.upgrades = []
        self.particles = []
        self.stars = []
        self.pirate_spawn_timer = 0
        
        self.base_pirate_speed = 1.5
        self.base_pirate_fire_rate = 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_speed = self.PLAYER_BASE_SPEED
        self.player_fire_cooldown = 0
        self.last_shift_state = 0
        
        self.unlocked_weapons = ["BLASTER"]
        self.current_weapon_idx = 0

        self.pirates.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.upgrades.clear()
        self.particles.clear()
        
        self.pirate_spawn_timer = 0
        self.base_pirate_speed = 1.5
        self.base_pirate_fire_rate = 0.01

        self._init_stars()
        self._spawn_pirate_formation()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # --- 1. Handle Input & Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vector = pygame.Vector2(0, 0)
        if movement == 1: move_vector.y = -1 # Up
        elif movement == 2: move_vector.y = 1 # Down
        elif movement == 3: move_vector.x = -1 # Left
        elif movement == 4: move_vector.x = 1 # Right
        
        if move_vector.length() > 0:
            move_vector.normalize_ip()
            self.player_pos += move_vector * self.player_speed
            self._wrap_around(self.player_pos)

        # Weapon Cycling (on press, not hold)
        if shift_held and not self.last_shift_state:
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.unlocked_weapons)
            # sfx: weapon_cycle.wav
        self.last_shift_state = shift_held

        # Firing
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)
        if space_held and self.player_fire_cooldown == 0:
            self._fire_weapon()
            # sfx: laser_shoot.wav

        # --- 2. Update Game State ---
        self.steps += 1
        self._update_difficulty()
        self._update_stars()
        self._update_projectiles()
        self._update_pirates()
        self._update_upgrades()
        self._update_particles()
        
        # Spawning
        self.pirate_spawn_timer -= 1
        if self.pirate_spawn_timer <= 0:
            self._spawn_pirate_formation()
            self.pirate_spawn_timer = self.PIRATE_SPAWN_INTERVAL
        
        # --- 3. Collisions and Rewards ---
        # Player projectiles vs Pirates
        for proj in self.player_projectiles[:]:
            for pirate in self.pirates[:]:
                if proj['pos'].distance_to(pirate['pos']) < pirate['size']:
                    pirate['health'] -= proj['damage']
                    reward += 0.1
                    self.player_projectiles.remove(proj)
                    if pirate['health'] <= 0:
                        reward += 1.0
                        self.score += pirate['score_value']
                        self._create_explosion(pirate['pos'], pirate['color'], 30)
                        if self.np_random.random() < 0.2: # 20% chance to drop upgrade
                            self._spawn_upgrade(pirate['pos'])
                        self.pirates.remove(pirate)
                        # sfx: explosion.wav
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj['pos'].distance_to(self.player_pos) < 15: # Player size approx
                self.player_health -= proj['damage']
                reward -= 0.1
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 15)
                self.enemy_projectiles.remove(proj)
                # sfx: player_hit.wav
                break
        
        # Player vs Upgrades
        for upgrade in self.upgrades[:]:
            if self.player_pos.distance_to(upgrade['pos']) < 20:
                reward += 5.0
                self._apply_upgrade(upgrade['type'])
                self.upgrades.remove(upgrade)
                # sfx: powerup.wav
                break
        
        # --- 4. Termination Conditions ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            terminated = True
            reward -= 50 # Penalty for dying
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            # sfx: player_explosion.wav
        elif self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100 # Bonus for winning
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_upgrades()
        self._render_projectiles()
        self._render_pirates()
        if self.player_health > 0:
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Update Methods ---
    def _update_difficulty(self):
        self.base_pirate_speed = 1.5 + (self.steps // 500) * 0.05
        self.base_pirate_fire_rate = 0.01 + (self.steps // 1000) * 0.02

    def _update_projectiles(self):
        for proj in self.player_projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.player_projectiles.remove(proj)
        for proj in self.enemy_projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().collidepoint(proj['pos']):
                self.enemy_projectiles.remove(proj)

    def _update_pirates(self):
        for pirate in self.pirates:
            pirate['pos'] += pirate['vel']
            if pirate['pos'].y > self.SCREEN_HEIGHT + pirate['size']:
                pirate['pos'].y = -pirate['size']
            if pirate['pos'].x < -pirate['size']:
                pirate['pos'].x = self.SCREEN_WIDTH + pirate['size']
            if pirate['pos'].x > self.SCREEN_WIDTH + pirate['size']:
                pirate['pos'].x = -pirate['size']
            
            if self.np_random.random() < self.base_pirate_fire_rate:
                self._pirate_fire(pirate)

    def _update_upgrades(self):
        for upgrade in self.upgrades:
            upgrade['pos'].y += 1
            if upgrade['pos'].y > self.SCREEN_HEIGHT + 10:
                self.upgrades.remove(upgrade)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_stars(self):
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.SCREEN_HEIGHT:
                star['pos'].y = 0
                star['pos'].x = self.np_random.uniform(0, self.SCREEN_WIDTH)

    # --- Spawning and Creation ---
    def _init_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), 
                                      self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'speed': self.np_random.uniform(0.2, 1.5),
                'radius': self.np_random.uniform(0.5, 1.5)
            })

    def _spawn_pirate_formation(self):
        formation_type = self.np_random.choice(['line', 'v', 'cluster'])
        start_x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
        
        if formation_type == 'line':
            for i in range(4):
                self._spawn_pirate('scout', pygame.Vector2(start_x + (i - 1.5) * 50, -30))
        elif formation_type == 'v':
            self._spawn_pirate('fighter', pygame.Vector2(start_x, -30))
            self._spawn_pirate('scout', pygame.Vector2(start_x - 40, -50))
            self._spawn_pirate('scout', pygame.Vector2(start_x + 40, -50))
        elif formation_type == 'cluster':
            for _ in range(self.np_random.integers(1, 3)):
                 self._spawn_pirate('bruiser', pygame.Vector2(start_x + self.np_random.uniform(-50, 50), -30))

    def _spawn_pirate(self, p_type, pos):
        if p_type == 'scout':
            pirate = {'type': 'scout', 'pos': pos, 'vel': pygame.Vector2(0, self.base_pirate_speed), 'health': 20, 'max_health': 20, 'size': 12, 'color': self.COLOR_PIRATE_SCOUT, 'score_value': 10}
        elif p_type == 'fighter':
            pirate = {'type': 'fighter', 'pos': pos, 'vel': pygame.Vector2(self.np_random.choice([-1, 1]), self.base_pirate_speed * 0.8), 'health': 40, 'max_health': 40, 'size': 15, 'color': self.COLOR_PIRATE_FIGHTER, 'score_value': 25}
        else: # bruiser
            pirate = {'type': 'bruiser', 'pos': pos, 'vel': pygame.Vector2(0, self.base_pirate_speed * 0.6), 'health': 80, 'max_health': 80, 'size': 18, 'color': self.COLOR_PIRATE_BRUISER, 'score_value': 50}
        self.pirates.append(pirate)

    def _spawn_upgrade(self, pos):
        upgrade_type = self.np_random.choice(['speed', 'firepower', 'weapon'])
        self.upgrades.append({'pos': pos.copy(), 'type': upgrade_type})

    def _fire_weapon(self):
        weapon_name = self.unlocked_weapons[self.current_weapon_idx]
        weapon_stats = self.WEAPONS[weapon_name]
        
        self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN * weapon_stats['cooldown_mod']
        
        proj_vel = pygame.Vector2(0, -weapon_stats['speed'])
        proj_pos = self.player_pos + pygame.Vector2(0, -20)
        
        self.player_projectiles.append({
            'pos': proj_pos,
            'vel': proj_vel,
            'color': weapon_stats['color'],
            'damage': weapon_stats['damage'],
            'radius': 4 if weapon_name != "PLASMA" else 6
        })
        # Muzzle flash
        self._create_explosion(self.player_pos + pygame.Vector2(0, -20), weapon_stats['color'], 5, 3)

    def _pirate_fire(self, pirate):
        if self.player_health <= 0: return
        direction = (self.player_pos - pirate['pos']).normalize()
        self.enemy_projectiles.append({
            'pos': pirate['pos'].copy(),
            'vel': direction * 4,
            'color': pirate['color'],
            'damage': 5,
            'radius': 3
        })

    def _create_explosion(self, pos, color, num_particles, life=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _apply_upgrade(self, u_type):
        if u_type == 'speed':
            self.player_speed = min(10, self.player_speed + 0.5)
        elif u_type == 'firepower':
            self.PLAYER_FIRE_COOLDOWN = max(3, self.PLAYER_FIRE_COOLDOWN - 1)
        elif u_type == 'weapon':
            available = list(self.WEAPONS.keys())
            if len(self.unlocked_weapons) < len(available):
                for w in available:
                    if w not in self.unlocked_weapons:
                        self.unlocked_weapons.append(w)
                        break

    # --- Rendering Methods ---
    def _render_stars(self):
        for star in self.stars:
            color_val = int(255 * (star['speed'] / 1.5))
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), int(star['radius']))

    def _render_player(self):
        p = self.player_pos
        points = [(p.x, p.y - 15), (p.x - 10, p.y + 10), (p.x + 10, p.y + 10)]
        glow_points = [(p.x, p.y - 18), (p.x - 13, p.y + 13), (p.x + 13, p.y + 13)]
        
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_pirates(self):
        for pirate in self.pirates:
            pos_int = (int(pirate['pos'].x), int(pirate['pos'].y))
            if pirate['type'] == 'scout':
                points = [(pos_int[0], pos_int[1] - 12), (pos_int[0] - 10, pos_int[1]), (pos_int[0], pos_int[1] + 12), (pos_int[0] + 10, pos_int[1])]
            elif pirate['type'] == 'fighter':
                points = [(pos_int[0] + 15, pos_int[1]), (pos_int[0] - 15, pos_int[1] - 10), (pos_int[0] - 10, pos_int[1]), (pos_int[0] - 15, pos_int[1] + 10)]
            else: # bruiser
                points = []
                for i in range(6):
                    angle = i * (2 * math.pi / 6)
                    points.append((pos_int[0] + 18 * math.cos(angle), pos_int[1] + 18 * math.sin(angle)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, pirate['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, pirate['color'])
            
            # Health bar
            health_pct = pirate['health'] / pirate['max_health']
            bar_y = pirate['pos'].y - pirate['size'] - 5
            pygame.draw.rect(self.screen, (50,0,0), (pirate['pos'].x - 15, bar_y, 30, 4))
            pygame.draw.rect(self.screen, (0,200,50), (pirate['pos'].x - 15, bar_y, 30 * health_pct, 4))

    def _render_projectiles(self):
        for proj in self.player_projectiles + self.enemy_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj['radius'], proj['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], proj['radius'], proj['color'])
    
    def _render_upgrades(self):
        for upgrade in self.upgrades:
            pos = (int(upgrade['pos'].x), int(upgrade['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_UPGRADE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_UPGRADE)
            char = upgrade['type'][0].upper()
            text = self.font_ui.render(char, True, self.COLOR_BG)
            self.screen.blit(text, (pos[0] - text.get_width() / 2, pos[1] - text.get_height() / 2))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:05d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, self.SCREEN_HEIGHT - 30, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (12, self.SCREEN_HEIGHT - 28, bar_width - 4, 16))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (12, self.SCREEN_HEIGHT - 28, (bar_width - 4) * health_pct, 16))

        # Weapon Display
        weapon_name = self.unlocked_weapons[self.current_weapon_idx]
        weapon_stats = self.WEAPONS[weapon_name]
        weapon_text = self.font_weapon.render(f"WEAPON: {weapon_name}", True, self.COLOR_UI_TEXT)
        text_rect = weapon_text.get_rect(bottomright=(self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 10))
        self.screen.blit(weapon_text, text_rect)
        
        # Game Over Text
        if self.game_over:
            end_text_str = "MISSION FAILED" if self.player_health <= 0 else "MISSION COMPLETE"
            end_font = pygame.font.SysFont("monospace", 48, bold=True)
            end_text = end_font.render(end_text_str, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _wrap_around(self, pos):
        if pos.x < 0: pos.x = self.SCREEN_WIDTH
        if pos.x > self.SCREEN_WIDTH: pos.x = 0
        if pos.y < 0: pos.y = self.SCREEN_HEIGHT
        if pos.y > self.SCREEN_HEIGHT: pos.y = 0

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # You must unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Galactic Conquest")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print info for debugging
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.TARGET_FPS)
        
        if terminated or truncated:
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()