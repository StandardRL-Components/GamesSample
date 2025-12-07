import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:34:39.138027
# Source Brief: brief_01145.md
# Brief Index: 1145
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Evade the police in a futuristic cityscape. Use clones and portals to outsmart your "
        "pursuers and reach the escape point before you're caught."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Space to create or use a portal, and "
        "press Shift to create a temporary clone to distract police."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    WORLD_SCALE = 2.5
    WORLD_WIDTH, WORLD_HEIGHT = int(WIDTH * WORLD_SCALE), int(HEIGHT * WORLD_SCALE)
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (10, 0, 25)
    COLOR_GRID = (30, 10, 60)
    COLOR_BUILDING = (20, 5, 45)

    COLOR_PLAYER = (0, 191, 255)  # DeepSkyBlue
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)

    COLOR_POLICE = (255, 0, 100)
    COLOR_POLICE_GLOW = (255, 0, 100, 50)

    COLOR_PORTAL = (50, 255, 50)
    COLOR_PORTAL_GLOW = (50, 255, 50, 50)

    COLOR_CLONE = (170, 0, 255)
    COLOR_CLONE_GLOW = (170, 0, 255, 50)

    COLOR_POWERUP = (255, 255, 0)
    COLOR_POWERUP_GLOW = (255, 255, 0, 50)

    COLOR_ESCAPE = (0, 255, 127) # SpringGreen
    COLOR_ESCAPE_GLOW = (0, 255, 127, 50)

    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PROXIMITY_WARN = (255, 255, 0)
    COLOR_PROXIMITY_ALERT = (255, 0, 0)

    # Game Parameters
    PLAYER_SPEED = 7.0
    PLAYER_RADIUS = 12
    POLICE_RADIUS = 12
    POLICE_CATCH_RADIUS = PLAYER_RADIUS + POLICE_RADIUS - 5
    POLICE_SIGHT_RADIUS = 400
    POLICE_COUNT = 4
    CLONE_LIFESPAN = 150  # 5 seconds at 30 FPS
    CLONE_COOLDOWN = 300 # 10 seconds
    PORTAL_COOLDOWN = 150 # 5 seconds
    POWERUP_COUNT = 3
    POWERUP_DURATION = 150 # 5 seconds

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.render_mode = render_mode
        self.camera_pos = np.zeros(2, dtype=np.float32)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.police = []
        self.clones = []
        self.particles = []
        self.powerups = []
        self.portal_entry = None
        self.escape_point = {}
        
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player = {
            'pos': np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float32),
            'speed_boost_timer': 0,
            'invisibility_timer': 0
        }
        
        # Debounce flags
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        # Cooldowns
        self.player['clone_cooldown'] = 0
        self.player['portal_cooldown'] = 0

        # Police state
        self.police = []
        for _ in range(self.POLICE_COUNT):
            self._spawn_police()

        # Other entities
        self.clones = []
        self.particles = []
        self.powerups = []
        for _ in range(self.POWERUP_COUNT):
            self._spawn_powerup()
        
        self.portal_entry = None

        # Escape point
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.WORLD_WIDTH * 0.4, self.WORLD_WIDTH * 0.45)
        self.escape_point = {
            'pos': self.player['pos'] + np.array([math.cos(angle) * dist, math.sin(angle) * dist])
        }
        self._clamp_to_world(self.escape_point['pos'])
        
        self.last_closest_police_dist = self._get_closest_police_dist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        
        # --- Handle Cooldowns & Power-ups ---
        if self.player['clone_cooldown'] > 0: self.player['clone_cooldown'] -= 1
        if self.player['portal_cooldown'] > 0: self.player['portal_cooldown'] -= 1
        if self.player['speed_boost_timer'] > 0: self.player['speed_boost_timer'] -= 1
        if self.player['invisibility_timer'] > 0: self.player['invisibility_timer'] -= 1

        # --- Handle Player Actions ---
        # Movement
        player_speed = self.PLAYER_SPEED * (1.5 if self.player['speed_boost_timer'] > 0 else 1.0)
        if movement == 1: self.player['pos'][1] -= player_speed  # Up
        elif movement == 2: self.player['pos'][1] += player_speed  # Down
        elif movement == 3: self.player['pos'][0] -= player_speed  # Left
        elif movement == 4: self.player['pos'][0] += player_speed  # Right
        self._clamp_to_world(self.player['pos'])
        self._create_particle(self.player['pos'], self.COLOR_PLAYER, 20) # Player trail

        # Action: Create Portal / Teleport
        is_space_press = space_held and not self.space_pressed_last_frame
        if is_space_press and self.player['portal_cooldown'] <= 0:
            if self.portal_entry is None:
                self.portal_entry = {'pos': self.player['pos'].copy(), 'created_step': self.steps}
                # sfx: portal_open
            else:
                teleport_target = self.portal_entry['pos'].copy()
                self.portal_entry = None # Consume portal
                self.player['pos'] = teleport_target
                self.player['portal_cooldown'] = self.PORTAL_COOLDOWN
                reward += 5.0
                # sfx: teleport
        self.space_pressed_last_frame = space_held

        # Action: Create Clone
        is_shift_press = shift_held and not self.shift_pressed_last_frame
        if is_shift_press and self.player['clone_cooldown'] <= 0:
            self.clones.append({
                'pos': self.player['pos'].copy(),
                'lifespan': self.CLONE_LIFESPAN
            })
            self.player['clone_cooldown'] = self.CLONE_COOLDOWN
            # sfx: clone_created
        self.shift_pressed_last_frame = shift_held
        
        # --- Update Game Entities ---
        self._update_police(reward)
        self._update_clones()
        self._update_particles()
        
        # --- Check Collisions & Collections ---
        # Power-ups
        for i in range(len(self.powerups) - 1, -1, -1):
            powerup = self.powerups[i]
            if self._dist(self.player['pos'], powerup['pos']) < self.PLAYER_RADIUS + 10:
                if powerup['type'] == 'speed': self.player['speed_boost_timer'] = self.POWERUP_DURATION
                elif powerup['type'] == 'invisibility': self.player['invisibility_timer'] = self.POWERUP_DURATION
                reward += 1.0
                self.powerups.pop(i)
                self._spawn_powerup()
                # sfx: powerup_collect

        # --- Calculate Continuous Reward ---
        closest_police_dist = self._get_closest_police_dist()
        if closest_police_dist > self.last_closest_police_dist:
            reward += 0.1
        self.last_closest_police_dist = closest_police_dist
        
        # --- Check Termination Conditions ---
        terminated = False
        # 1. Caught by police
        for p in self.police:
            if self._dist(self.player['pos'], p['pos']) < self.POLICE_CATCH_RADIUS:
                reward = -100.0
                terminated = True
                self.game_over = True
                # sfx: caught
                break
        
        # 2. Reached escape point
        if not terminated and self._dist(self.player['pos'], self.escape_point['pos']) < self.PLAYER_RADIUS + 15:
            reward = 100.0
            terminated = True
            self.game_over = True
            # sfx: win
        
        # 3. Max steps
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_police(self, reward):
        base_speed = 4.0 + (self.steps / 200) * 0.05
        
        distracted_police_this_frame = set()

        for i, p in enumerate(self.police):
            target = self.player['pos']
            target_is_clone = False

            # If player is invisible, police move to last known location then stop
            if self.player['invisibility_timer'] > 0:
                if self._dist(p['pos'], p['last_seen_pos']) > 5:
                    target = p['last_seen_pos']
                else: # Reached last known pos, so stop
                    self._create_particle(p['pos'], self.COLOR_POLICE, 20)
                    continue
            else:
                 p['last_seen_pos'] = self.player['pos'].copy()


            # Check if a clone is a more attractive target
            player_dist = self._dist(p['pos'], self.player['pos'])
            
            if self.player['invisibility_timer'] <= 0:
                for j, clone in enumerate(self.clones):
                    clone_dist = self._dist(p['pos'], clone['pos'])
                    if clone_dist < player_dist:
                        player_dist = clone_dist
                        target = clone['pos']
                        target_is_clone = True
                        if j not in p['distracted_by_clone']:
                            reward += 2.0
                            p['distracted_by_clone'].add(j)
                            distracted_police_this_frame.add(i)

            # Move towards target
            if self._dist(p['pos'], target) > 1:
                direction = (target - p['pos']) / self._dist(p['pos'], target)
                p['pos'] += direction * base_speed
            
            self._clamp_to_world(p['pos'])
            self._create_particle(p['pos'], self.COLOR_POLICE, 20)
        
        # Clear distraction memory for clones that are gone
        active_clone_indices = set(range(len(self.clones)))
        for p in self.police:
            p['distracted_by_clone'] &= active_clone_indices

    def _update_clones(self):
        for i in range(len(self.clones) - 1, -1, -1):
            self.clones[i]['lifespan'] -= 1
            if self.clones[i]['lifespan'] <= 0:
                self.clones.pop(i)

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - 0.5)
            if p['lifespan'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.camera_pos = self.player['pos'] - np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_background()

        # Render entities (back to front)
        self._draw_entity(self.escape_point, self.COLOR_ESCAPE, self.COLOR_ESCAPE_GLOW, 15, is_pulsating=True)
        if self.portal_entry:
            self._draw_entity(self.portal_entry, self.COLOR_PORTAL, self.COLOR_PORTAL_GLOW, 15, is_shimmering=True)
        
        for powerup in self.powerups:
            self._draw_entity(powerup, self.COLOR_POWERUP, self.COLOR_POWERUP_GLOW, 10, is_pulsating=True)

        for particle in self.particles:
            self._draw_particle(particle)

        for clone in self.clones:
            self._draw_clone(clone)

        for p in self.police:
            self._draw_vehicle(p, self.COLOR_POLICE, self.COLOR_POLICE_GLOW, self.POLICE_RADIUS)
        
        if self.player['invisibility_timer'] <= 0:
            self._draw_vehicle(self.player, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.PLAYER_RADIUS)
        else: # Draw faint outline when invisible
            pos = self.player['pos'] - self.camera_pos
            alpha = 50 + math.sin(self.steps * 0.5) * 20
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.PLAYER_RADIUS, (*self.COLOR_PLAYER, int(alpha)))


    def _render_ui(self):
        # Proximity warning
        closest_dist = self._get_closest_police_dist()
        if closest_dist < self.POLICE_SIGHT_RADIUS / 1.5:
            warn_alpha = int(255 * (1 - (closest_dist / (self.POLICE_SIGHT_RADIUS / 1.5))))
            if closest_dist < self.POLICE_SIGHT_RADIUS / 4:
                color = self.COLOR_PROXIMITY_ALERT
            else:
                color = self.COLOR_PROXIMITY_WARN
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, warn_alpha), (0, 0, self.WIDTH, self.HEIGHT), 10)
            self.screen.blit(s, (0,0))

        # Text UI
        dist_to_escape = self._dist(self.player['pos'], self.escape_point['pos'])
        dist_text = self.font_large.render(f"ESCAPE: {int(dist_to_escape)}m", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (10, 10))

        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Cooldowns UI
        def draw_cooldown_bar(y, label, cooldown, max_cooldown, color):
            if cooldown > 0:
                ratio = cooldown / max_cooldown
                bar_width = 100
                pygame.draw.rect(self.screen, (50,50,50), (10, y, bar_width, 15))
                pygame.draw.rect(self.screen, color, (10, y, bar_width * ratio, 15))
                text = self.font_small.render(label, True, self.COLOR_UI_TEXT)
                self.screen.blit(text, (15 + bar_width, y))

        draw_cooldown_bar(self.HEIGHT - 45, "CLONE", self.player['clone_cooldown'], self.CLONE_COOLDOWN, self.COLOR_CLONE)
        draw_cooldown_bar(self.HEIGHT - 25, "PORTAL", self.player['portal_cooldown'], self.PORTAL_COOLDOWN, self.COLOR_PORTAL)

    def _draw_background(self):
        cam_x, cam_y = self.camera_pos
        grid_size = 50
        
        start_x = -int(cam_x % grid_size)
        start_y = -int(cam_y % grid_size)

        for x in range(start_x, self.WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(start_y, self.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_glow_circle(self, pos, color, glow_color, radius):
        x, y = int(pos[0]), int(pos[1])
        # Draw multiple layers for a bloom effect
        for i in range(4, 0, -1):
            alpha = glow_color[3] // i
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius * (1 + i * 0.2)), (*glow_color[:3], alpha))
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _draw_vehicle(self, entity, color, glow_color, radius):
        pos = entity['pos'] - self.camera_pos
        self._draw_glow_circle(pos, color, glow_color, radius)

    def _draw_clone(self, clone):
        pos = clone['pos'] - self.camera_pos
        alpha = int(150 * (clone['lifespan'] / self.CLONE_LIFESPAN))
        color = (*self.COLOR_CLONE, alpha)
        glow_color = (*self.COLOR_CLONE_GLOW[:3], int(alpha*0.5))
        self._draw_glow_circle(pos, color, glow_color, self.PLAYER_RADIUS)

    def _draw_entity(self, entity, color, glow_color, radius, is_pulsating=False, is_shimmering=False):
        pos = entity['pos'] - self.camera_pos
        current_radius = radius
        if is_pulsating:
            current_radius = radius * (1 + 0.15 * math.sin(self.steps * 0.1))
        if is_shimmering:
            pos += self.np_random.uniform(-1, 1, 2)
        self._draw_glow_circle(pos, color, glow_color, int(current_radius))

    def _draw_particle(self, p):
        pos = p['pos'] - self.camera_pos
        alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
        color = (*p['color'][:3], alpha)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(p['radius']), color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_closest_police_dist(self):
        if not self.police: return float('inf')
        return min(self._dist(self.player['pos'], p['pos']) for p in self.police)

    def _spawn_police(self):
        while True:
            pos = self.np_random.uniform([0, 0], [self.WORLD_WIDTH, self.WORLD_HEIGHT])
            if self._dist(pos, self.player['pos']) > self.POLICE_SIGHT_RADIUS:
                self.police.append({
                    'pos': pos.astype(np.float32),
                    'last_seen_pos': self.player['pos'].copy(),
                    'distracted_by_clone': set()
                })
                break

    def _spawn_powerup(self):
        pos = self.np_random.uniform([0, 0], [self.WORLD_WIDTH, self.WORLD_HEIGHT])
        ptype = self.np_random.choice(['speed', 'invisibility'])
        self.powerups.append({'pos': pos.astype(np.float32), 'type': ptype})

    def _create_particle(self, pos, color, lifespan):
        self.particles.append({
            'pos': pos.copy() + self.np_random.uniform(-3, 3, 2),
            'color': color,
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'radius': self.np_random.uniform(3, 6)
        })

    def _clamp_to_world(self, pos):
        pos[0] = np.clip(pos[0], 0, self.WORLD_WIDTH)
        pos[1] = np.clip(pos[1], 0, self.WORLD_HEIGHT)
    
    def _dist(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")

    obs, info = env.reset()
    done = False
    
    # Use a window to display the game
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cyberbike Escape")
    clock = pygame.time.Clock()

    total_reward = 0
    total_steps = 0

    movement_keys = {
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2,
        pygame.K_LEFT: 3, pygame.K_a: 3,
        pygame.K_RIGHT: 4, pygame.K_d: 4
    }

    running = True
    while running:
        mov_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, action in movement_keys.items():
            if keys[key]:
                mov_action = action
                break # Prioritize first key in dict order
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [mov_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_steps += 1

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)
        
        if done:
            print(f"Episode finished after {total_steps} steps.")
            print(f"Final Score: {total_reward:.2f}")
            # Reset for a new game
            obs, info = env.reset()
            done = False
            total_reward = 0
            total_steps = 0
            pygame.time.wait(1000) # Pause for a second before restarting


    env.close()