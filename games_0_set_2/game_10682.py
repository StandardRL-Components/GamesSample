import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:53:05.072086
# Source Brief: brief_00682.md
# Brief Index: 682
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a "Missile Command" style game.

    The player must defend their cities from incoming enemy missiles by firing
    their own standard and nuke missiles. The goal is to survive for a fixed
    duration. The game features retro arcade visuals with procedural terrain,
    particle effects, and smooth animations.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Crosshair Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Fire Standard Missile (0=released, 1=pressed)
    - actions[2]: Fire Nuke Missile (0=released, 1=pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your cities from incoming enemy missiles. Use your crosshair to launch "
        "counter-missiles and survive until the timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the crosshair. "
        "Press space to fire a standard missile and shift to fire a nuke."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_STANDARD_MISSILES = 25
        self.INITIAL_NUKE_MISSILES = 3
        self.NUM_CITIES = 4
        self.GROUND_LEVEL = self.HEIGHT - 40
        self.CROSSHAIR_SPEED = 20
        self.PLAYER_MISSILE_SPEED = 8
        
        # --- Colors (High Contrast, Retro Arcade) ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_STARS = (200, 200, 220)
        self.COLOR_GROUND = (40, 30, 20)
        self.COLOR_CITY_ALIVE = (0, 150, 255)
        self.COLOR_CITY_GLOW = (100, 200, 255, 50)
        self.COLOR_CITY_DESTROYED = (60, 60, 60)
        self.COLOR_CROSSHAIR = (0, 255, 255)
        self.COLOR_CROSSHAIR_GLOW = (0, 255, 255, 100)
        self.COLOR_PLAYER_MISSILE = (0, 255, 0)
        self.COLOR_ENEMY_MISSILE = (255, 50, 50)
        self.COLOR_EXPLOSION_PROFILE_STANDARD = [(255, 255, 255), (255, 255, 0), (255, 128, 0)]
        self.COLOR_EXPLOSION_PROFILE_NUKE = [(255, 255, 255), (0, 255, 255), (0, 128, 255)]
        self.COLOR_UI_TEXT = (255, 255, 220)
        
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
        self.font_ui = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_game_over = pygame.font.SysFont('monospace', 48, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crosshair_pos = [0, 0]
        self.cities = []
        self.ground_poly = []
        self.stars = []
        self.enemy_missiles = []
        self.player_missiles = []
        self.explosions = []
        self.standard_missile_count = 0
        self.nuke_missile_count = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.enemy_spawn_rate = 0.0
        self.enemy_speed = 0.0

        # Initialize state
        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.crosshair_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        
        self.standard_missile_count = self.INITIAL_STANDARD_MISSILES
        self.nuke_missile_count = self.INITIAL_NUKE_MISSILES
        
        self.enemy_missiles.clear()
        self.player_missiles.clear()
        self.explosions.clear()
        
        self.last_space_held = False
        self.last_shift_held = False

        # Procedurally generate terrain and cities
        self._generate_world()
        
        return self._get_observation(), self._get_info()

    def _generate_world(self):
        # Generate stars
        self.stars = [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.GROUND_LEVEL)) for _ in range(150)]
        
        # Generate terrain
        points = []
        x = 0
        while x <= self.WIDTH:
            y = self.np_random.uniform(self.GROUND_LEVEL, self.HEIGHT - 10)
            points.append((x, y))
            x += self.np_random.uniform(20, 80)
        points.append((self.WIDTH, points[-1][1]))
        self.ground_poly = [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)] + points
        
        # Generate cities
        self.cities.clear()
        city_width = 30
        city_height = 15
        spacing = (self.WIDTH - self.NUM_CITIES * city_width) / (self.NUM_CITIES + 1)
        for i in range(self.NUM_CITIES):
            x = spacing * (i + 1) + city_width * i
            y = self.GROUND_LEVEL - city_height
            self.cities.append({'rect': pygame.Rect(x, y, city_width, city_height), 'alive': True})
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        
        # 1. Handle player input
        self._handle_input(action)
        
        # 2. Update game state
        self._spawn_enemies()
        self._update_player_missiles()
        reward += self._update_enemy_missiles()
        reward += self._update_and_collide_explosions()

        # 3. Check for termination
        terminated = False
        num_alive_cities = sum(1 for city in self.cities if city['alive'])
        if num_alive_cities == 0:
            terminated = True
            # No extra penalty here, city loss is penalized individually
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if num_alive_cities > 0:
                reward += 100.0 # Victory bonus
                
        if terminated:
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # Move crosshair
        if movement == 1: self.crosshair_pos[1] -= self.CROSSHAIR_SPEED  # Up
        elif movement == 2: self.crosshair_pos[1] += self.CROSSHAIR_SPEED # Down
        elif movement == 3: self.crosshair_pos[0] -= self.CROSSHAIR_SPEED # Left
        elif movement == 4: self.crosshair_pos[0] += self.CROSSHAIR_SPEED # Right
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.GROUND_LEVEL)

        # Fire standard missile
        if space_pressed and not self.last_space_held and self.standard_missile_count > 0:
            self.standard_missile_count -= 1
            self._fire_player_missile('standard')
            # SFX: Player_Fire_Standard.wav

        # Fire nuke missile
        if shift_pressed and not self.last_shift_held and self.nuke_missile_count > 0:
            self.nuke_missile_count -= 1
            self._fire_player_missile('nuke')
            # SFX: Player_Fire_Nuke.wav

        self.last_space_held = space_pressed
        self.last_shift_held = shift_pressed

    def _fire_player_missile(self, missile_type):
        start_pos = [self.WIDTH / 2, self.GROUND_LEVEL]
        target_pos = list(self.crosshair_pos)
        
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        dist = math.hypot(dx, dy)
        if dist == 0: return
        
        vel = [dx / dist * self.PLAYER_MISSILE_SPEED, dy / dist * self.PLAYER_MISSILE_SPEED]
        
        self.player_missiles.append({
            'pos': start_pos, 'vel': vel, 'target': target_pos,
            'type': missile_type, 'trail': [start_pos.copy() for _ in range(5)]
        })
        
    def _spawn_enemies(self):
        # Difficulty scaling
        self.enemy_spawn_rate = min(0.5, 0.1 + self.steps * 0.001)
        self.enemy_speed = min(2.0, 1.0 + (self.steps // 100) * 0.01)

        if self.np_random.random() < self.enemy_spawn_rate:
            alive_cities = [c for c in self.cities if c['alive']]
            if not alive_cities: return # No targets left

            target_city = self.np_random.choice(alive_cities)
            start_x = self.np_random.uniform(0, self.WIDTH)
            start_pos = [start_x, 0]
            target_pos = [target_city['rect'].centerx, target_city['rect'].centery]
            
            # Bezier curve control point for arcing trajectory
            control_x = (start_x + target_pos[0]) / 2 + self.np_random.uniform(-150, 150)
            control_y = min(start_pos[1], target_pos[1]) - self.np_random.uniform(50, 200)
            control_pos = [control_x, control_y]
            
            self.enemy_missiles.append({
                'p0': start_pos, 'p1': control_pos, 'p2': target_pos,
                'progress': 0.0, 'speed_factor': self.enemy_speed,
                'trail': [start_pos.copy() for _ in range(5)]
            })

    def _update_player_missiles(self):
        for m in self.player_missiles[:]:
            m['pos'][0] += m['vel'][0]
            m['pos'][1] += m['vel'][1]
            
            # Update trail
            m['trail'].pop(0)
            m['trail'].append(m['pos'].copy())
            
            # Check for target arrival
            if math.hypot(m['pos'][0] - m['target'][0], m['pos'][1] - m['target'][1]) < self.PLAYER_MISSILE_SPEED:
                self._create_explosion(m['pos'], m['type'])
                self.player_missiles.remove(m)

    def _update_enemy_missiles(self):
        reward = 0
        for m in self.enemy_missiles[:]:
            # Update progress based on speed
            dist = math.hypot(m['p2'][0] - m['p0'][0], m['p2'][1] - m['p0'][1])
            increment = (1 / (dist / m['speed_factor'])) if dist > 0 else 1.0
            m['progress'] = min(1.0, m['progress'] + increment)

            # Bezier curve calculation for position
            t = m['progress']
            pos_x = (1-t)**2 * m['p0'][0] + 2*(1-t)*t * m['p1'][0] + t**2 * m['p2'][0]
            pos_y = (1-t)**2 * m['p0'][1] + 2*(1-t)*t * m['p1'][1] + t**2 * m['p2'][1]
            current_pos = [pos_x, pos_y]
            
            # Update trail
            m['trail'].pop(0)
            m['trail'].append(current_pos)

            # Check for impact
            if m['progress'] >= 1.0:
                self.enemy_missiles.remove(m)
                # Check city collision
                hit_city = False
                for city in self.cities:
                    if city['alive'] and city['rect'].collidepoint(current_pos):
                        city['alive'] = False
                        hit_city = True
                        reward -= 10.0
                        self._create_explosion(current_pos, 'impact')
                        # SFX: City_Destroyed.wav
                        break
                if not hit_city:
                    self._create_explosion(current_pos, 'impact_ground')
                    # SFX: Explosion_Small.wav
        return reward

    def _update_and_collide_explosions(self):
        reward = 0
        for e in self.explosions[:]:
            e['lifetime'] -= 1
            e['radius'] = e['max_radius'] * (1 - (e['lifetime'] / e['max_lifetime'])**2) # Ease-out expansion
            
            if e['lifetime'] <= 0:
                self.explosions.remove(e)
                continue
            
            # Collision with enemy missiles
            for m in self.enemy_missiles[:]:
                missile_pos = self._get_bezier_pos(m, m['progress'])
                if math.hypot(e['pos'][0] - missile_pos[0], e['pos'][1] - missile_pos[1]) < e['radius']:
                    self.enemy_missiles.remove(m)
                    reward += 0.1
                    e['kill_count'] += 1
                    if e['type'] == 'nuke' and e['kill_count'] > 1:
                        reward += 1.0 # Nuke multi-kill bonus
                    self._create_explosion(missile_pos, 'interception')
                    # SFX: Intercept_Success.wav
        return reward

    def _get_bezier_pos(self, missile, t):
        pos_x = (1-t)**2 * missile['p0'][0] + 2*(1-t)*t * missile['p1'][0] + t**2 * missile['p2'][0]
        pos_y = (1-t)**2 * missile['p0'][1] + 2*(1-t)*t * missile['p1'][1] + t**2 * missile['p2'][1]
        return [pos_x, pos_y]

    def _create_explosion(self, pos, type):
        if type == 'standard':
            params = {'max_radius': 40, 'max_lifetime': 25, 'profile': self.COLOR_EXPLOSION_PROFILE_STANDARD}
        elif type == 'nuke':
            params = {'max_radius': 80, 'max_lifetime': 40, 'profile': self.COLOR_EXPLOSION_PROFILE_NUKE}
        elif type == 'interception':
            params = {'max_radius': 15, 'max_lifetime': 15, 'profile': self.COLOR_EXPLOSION_PROFILE_STANDARD}
        else: # impact or impact_ground
            params = {'max_radius': 25, 'max_lifetime': 20, 'profile': self.COLOR_EXPLOSION_PROFILE_STANDARD}
        
        self.explosions.append({
            'pos': pos, 'radius': 0, 'kill_count': 0, 'type': type,
            'max_radius': params['max_radius'], 'lifetime': params['max_lifetime'],
            'max_lifetime': params['max_lifetime'], 'color_profile': params['profile']
        })
        # SFX: Explosion_Large.wav (for standard/nuke), Explosion_Small.wav (for others)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_cities()
        self._render_missiles()
        self._render_explosions()
        if not self.game_over:
            self._render_crosshair()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            self.screen.set_at(star, self.COLOR_STARS)
        pygame.gfxdraw.filled_polygon(self.screen, self.ground_poly, self.COLOR_GROUND)
        pygame.gfxdraw.aapolygon(self.screen, self.ground_poly, self.COLOR_GROUND)

    def _render_cities(self):
        for city in self.cities:
            if city['alive']:
                pygame.draw.rect(self.screen, self.COLOR_CITY_ALIVE, city['rect'])
                # Glow effect
                glow_rect = city['rect'].inflate(8, 8)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_CITY_GLOW, s.get_rect(), border_radius=4)
                self.screen.blit(s, glow_rect.topleft)
            else:
                pygame.draw.rect(self.screen, self.COLOR_CITY_DESTROYED, city['rect'])

    def _render_missiles(self):
        # Player missiles
        for m in self.player_missiles:
            self._render_trail(m['trail'], self.COLOR_PLAYER_MISSILE)
            pygame.gfxdraw.filled_circle(self.screen, int(m['pos'][0]), int(m['pos'][1]), 3, self.COLOR_PLAYER_MISSILE)
        # Enemy missiles
        for m in self.enemy_missiles:
            pos = self._get_bezier_pos(m, m['progress'])
            self._render_trail(m['trail'], self.COLOR_ENEMY_MISSILE)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 3, self.COLOR_ENEMY_MISSILE)
            
    def _render_trail(self, trail_points, color):
        for i, p in enumerate(trail_points):
            if i > 0:
                alpha = int(255 * (i / len(trail_points)))
                try:
                    pygame.draw.aaline(self.screen, color + (alpha,), trail_points[i-1], p, 1)
                except (TypeError, ValueError): # Handle potential color format issues
                    pygame.draw.aaline(self.screen, (*color[:3], alpha), trail_points[i-1], p, 1)

    def _render_explosions(self):
        for e in self.explosions:
            # Layered, colored circles for a better explosion effect
            r = e['radius']
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            c1, c2, c3 = e['color_profile']
            
            if r > 0: pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(r), c3)
            if r > 5: pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(r * 0.66), c2)
            if r > 8: pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(r * 0.33), c1)

    def _render_crosshair(self):
        x, y = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = 12
        glow_size = 18
        # Glow
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR_GLOW, (x - glow_size, y), (x + glow_size, y), 3)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR_GLOW, (x, y - glow_size), (x, y + glow_size), 3)
        # Main lines
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x - size, y), (x + size, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x, y - size), (x, y + size), 1)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Time
        time_surf = self.font_ui.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Ammo
        std_ammo_surf = self.font_ui.render(f"STD: {self.standard_missile_count}", True, self.COLOR_PLAYER_MISSILE)
        nuke_ammo_surf = self.font_ui.render(f"NUKE: {self.nuke_missile_count}", True, self.COLOR_CROSSHAIR)
        self.screen.blit(std_ammo_surf, (10, self.HEIGHT - 30))
        self.screen.blit(nuke_ammo_surf, (std_ammo_surf.get_width() + 30, self.HEIGHT - 30))

        if self.game_over:
            num_alive = sum(1 for c in self.cities if c['alive'])
            if self.steps >= self.MAX_STEPS and num_alive > 0:
                msg = "VICTORY"
            else:
                msg = "GAME OVER"
            
            over_surf = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            over_rect = over_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play loop to test the environment
    # This part of the code will not be run by the evaluator
    # but is useful for testing.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual testing
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Missile Command Gym Environment")
        clock = pygame.time.Clock()

        while not done:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

        # Keep the final screen visible for a moment
        if terminated:
            pygame.time.wait(2000)

    except pygame.error as e:
        print(f"Pygame error (likely due to headless mode): {e}")
        print("This is expected if you run this script without a display.")
    finally:
        env.close()