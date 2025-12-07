import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:18:12.303187
# Source Brief: brief_02544.md
# Brief Index: 2544
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a drone through a surreal dreamscape, placing defensive towers and manipulating time to fend off growing creatures as you race to the finish line."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the drone. Hold space to slow down time and press shift to place a defensive tower."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 6400
    FINISH_LINE_X = WORLD_WIDTH - 300
    MAX_STEPS = 5000
    FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_DRONE = (0, 150, 255)
    COLOR_DRONE_GLOW = (0, 75, 128)
    COLOR_CREATURE = (255, 50, 0)
    COLOR_CREATURE_GLOW = (128, 25, 0)
    COLOR_TOWER = (0, 255, 100)
    COLOR_TOWER_GLOW = (0, 128, 50)
    COLOR_BULLET = (150, 255, 150)
    COLOR_FINISH_LINE = (255, 0, 255)
    COLOR_FINISH_GLOW = (128, 0, 128)
    COLOR_TRACK_LINE = (100, 100, 200)
    COLOR_TRACK_GLOW = (50, 50, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIME_SLOW_VIGNETTE = (200, 200, 255)

    # Drone
    DRONE_START_SIZE = 18
    DRONE_MIN_SIZE = 5
    DRONE_SPEED = 8.0
    DRONE_SHRINK_RATE = 0.999 # Shrinks by 0.1% every 10 steps -> (1-0.001)^(1/10)
    DRONE_CAMERA_X_OFFSET = SCREEN_WIDTH / 4

    # Time Slow
    TIME_SLOW_MAX = 100.0
    TIME_SLOW_DRAIN_RATE = 1.0
    TIME_SLOW_REGEN_RATE = 0.3
    TIME_SLOW_MULTIPLIER = 0.3

    # Towers
    TOWER_COOLDOWN_STEPS = 15 # Can place a tower every 0.5s
    TOWER_RANGE = 150
    TOWER_FIRE_RATE = 20 # Fires every 20 steps
    BULLET_SPEED = 12.0
    BULLET_LIFETIME = 40

    # Creatures
    CREATURE_START_SIZE = 10
    CREATURE_SPAWN_RATE_INITIAL = 100
    CREATURE_SPAWN_RATE_MIN = 30
    CREATURE_SPAWN_RATE_SCALING_STEPS = 500
    CREATURE_BASE_SPEED = 1.5
    CREATURE_GROWTH_RATE_INITIAL = 1.001
    CREATURE_GROWTH_RATE_SCALING_STEPS = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game state variables
        self.drone_pos = None
        self.drone_size = None
        self.creatures = None
        self.towers = None
        self.bullets = None
        self.particles = None
        self.stars = None
        self.camera_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_slow_meter = None
        self.is_time_slowing = None
        self.tower_placement_cooldown = None
        self.creature_spawn_timer = None
        self.creature_spawn_rate = None
        self.creature_growth_rate = None
        self.prev_space_held = None
        self.prev_shift_held = None

        # Initialize state
        # A reset is called by the environment wrapper, so we don't need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.drone_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.drone_size = self.DRONE_START_SIZE
        self.creatures = []
        self.towers = []
        self.bullets = []
        self.particles = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_slow_meter = self.TIME_SLOW_MAX
        self.is_time_slowing = False
        self.tower_placement_cooldown = 0
        self.creature_spawn_rate = self.CREATURE_SPAWN_RATE_INITIAL
        self.creature_spawn_timer = self.creature_spawn_rate
        self.creature_growth_rate = self.CREATURE_GROWTH_RATE_INITIAL
        self.prev_space_held = False
        self.prev_shift_held = False

        # Create parallax starfield
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)]),
                'depth': self.np_random.uniform(0.1, 0.8), # Slower scrolling for more distant stars
                'size': self.np_random.integers(1, 3)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- TIME MANIPULATION ---
        time_just_activated = space_held and not self.prev_space_held
        if space_held and self.time_slow_meter > 0:
            self.is_time_slowing = True
            self.time_slow_meter = max(0, self.time_slow_meter - self.TIME_SLOW_DRAIN_RATE)
            if time_just_activated:
                reward += 5 # Reward for activating time slow
                # Sound: Time slow activate
                self._create_particles(self.drone_pos, 20, (200, 200, 255), 2, 4, 15, 'burst')
        else:
            self.is_time_slowing = False
            self.time_slow_meter = min(self.TIME_SLOW_MAX, self.time_slow_meter + self.TIME_SLOW_REGEN_RATE)
        
        game_speed_multiplier = self.TIME_SLOW_MULTIPLIER if self.is_time_slowing else 1.0

        # --- PLAYER ACTIONS ---
        # Drone movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1
        elif movement == 2: move_vec[1] += 1
        elif movement == 3: move_vec[0] -= 1
        elif movement == 4: move_vec[0] += 1
        
        if np.linalg.norm(move_vec) > 0:
            self.drone_pos += move_vec * self.DRONE_SPEED
            # Sound: Drone hum
            # Create thrust particles
            if self.steps % 2 == 0:
                p_pos = self.drone_pos - move_vec * self.drone_size
                self._create_particles(p_pos, 1, self.COLOR_DRONE, 1, 3, 10, 'trail', -move_vec * 2)

        # Clamp drone position
        self.drone_pos[0] = np.clip(self.drone_pos[0], self.camera_x, self.camera_x + self.SCREEN_WIDTH - self.drone_size)
        self.drone_pos[1] = np.clip(self.drone_pos[1], self.drone_size, self.SCREEN_HEIGHT - self.drone_size)

        # Tower placement (on button press)
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.tower_placement_cooldown <= 0:
            # Sound: Tower place
            self.towers.append({
                'pos': self.drone_pos.copy(),
                'fire_cooldown': self.np_random.integers(0, self.TOWER_FIRE_RATE), # Stagger initial fire
            })
            self.tower_placement_cooldown = self.TOWER_COOLDOWN_STEPS
            self._create_particles(self.drone_pos, 15, self.COLOR_TOWER, 2, 5, 20, 'burst')

        # --- GAME LOGIC UPDATES (affected by time slow) ---
        logic_steps = game_speed_multiplier
        while logic_steps > 0:
            step_delta = min(logic_steps, 1.0)
            
            # Update Towers and fire bullets
            for tower in self.towers:
                tower['fire_cooldown'] -= step_delta
                if tower['fire_cooldown'] <= 0:
                    target = self._find_closest_creature(tower['pos'], self.TOWER_RANGE)
                    if target:
                        # Sound: Tower shoot
                        direction = (target['pos'] - tower['pos'])
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                            self.bullets.append({
                                'pos': tower['pos'].copy(),
                                'vel': direction * self.BULLET_SPEED,
                                'lifetime': self.BULLET_LIFETIME
                            })
                            tower['fire_cooldown'] = self.TOWER_FIRE_RATE

            # Update Bullets
            for bullet in self.bullets:
                bullet['pos'] += bullet['vel'] * step_delta
                bullet['lifetime'] -= step_delta

            # Update Creatures
            for creature in self.creatures:
                direction = self.drone_pos - creature['pos']
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                creature['pos'] += direction * self.CREATURE_BASE_SPEED * step_delta
                creature['size'] *= (self.creature_growth_rate ** step_delta)

            logic_steps -= 1.0

        # --- COLLISION AND STATE CHECKS (full step) ---
        # Bullet-Creature collisions
        new_bullets = []
        for bullet in self.bullets:
            hit = False
            for creature in self.creatures:
                if np.linalg.norm(bullet['pos'] - creature['pos']) < creature['size']:
                    # Sound: Creature hit/destroy
                    self._create_particles(creature['pos'], 30, self.COLOR_CREATURE, 1, 4, 25, 'burst')
                    creature['alive'] = False
                    hit = True
                    reward += 1
                    self.score += 10
                    break 
            if not hit and bullet['lifetime'] > 0:
                new_bullets.append(bullet)
        self.bullets = new_bullets
        self.creatures = [c for c in self.creatures if c.get('alive', True)]

        # Creature-Drone collision
        for creature in self.creatures:
            if np.linalg.norm(self.drone_pos - creature['pos']) < self.drone_size + creature['size']:
                # Sound: Drone explosion
                self.game_over = True
                reward -= 100
                self._create_particles(self.drone_pos, 100, self.COLOR_DRONE, 2, 8, 40, 'burst')
                break

        # --- PER-STEP UPDATES (not affected by time slow) ---
        self.steps += 1
        reward += 0.01 # Small survival reward

        # Update cooldowns
        if self.tower_placement_cooldown > 0:
            self.tower_placement_cooldown -= 1

        # Update drone size
        if self.steps % 10 == 0:
            self.drone_size = max(self.DRONE_MIN_SIZE, self.drone_size * self.DRONE_SHRINK_RATE)

        # Update difficulty
        if self.steps > 0:
            if self.steps % self.CREATURE_GROWTH_RATE_SCALING_STEPS == 0:
                self.creature_growth_rate += 0.0005
            if self.steps % self.CREATURE_SPAWN_RATE_SCALING_STEPS == 0:
                self.creature_spawn_rate = max(self.CREATURE_SPAWN_RATE_MIN, self.creature_spawn_rate - 5)

        # Spawn new creatures
        self.creature_spawn_timer -= 1
        if self.creature_spawn_timer <= 0:
            spawn_y = self.np_random.uniform(self.CREATURE_START_SIZE, self.SCREEN_HEIGHT - self.CREATURE_START_SIZE)
            spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
            self.creatures.append({
                'pos': np.array([spawn_x, spawn_y]),
                'size': self.CREATURE_START_SIZE
            })
            self.creature_spawn_timer = self.creature_spawn_rate

        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        # Update camera
        self.camera_x = self.drone_pos[0] - self.DRONE_CAMERA_X_OFFSET
        
        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Check termination conditions
        truncated = False
        if self.game_over:
            terminated = True
        elif self.drone_pos[0] >= self.FINISH_LINE_X:
            terminated = True
            reward += 100
            self.score += 1000
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Using terminated as it's a time limit, not an arbitrary truncation
            # truncated = True # Alternatively, use truncated if it's a step limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _find_closest_creature(self, pos, max_range):
        closest_creature = None
        min_dist_sq = max_range ** 2
        for creature in self.creatures:
            dist_sq = np.sum((creature['pos'] - pos) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_creature = creature
        return closest_creature

    def _create_particles(self, pos, count, color, min_size, max_size, lifetime, p_type, base_vel=np.array([0,0])):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            elif p_type == 'trail':
                vel = base_vel + self.np_random.standard_normal(2) * 0.5
            
            self.particles.append({
                'pos': pos.copy() + self.np_random.standard_normal(2) * 2,
                'vel': vel,
                'lifetime': self.np_random.integers(lifetime // 2, lifetime + 1),
                'color': color,
                'size': self.np_random.uniform(min_size, max_size)
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- BACKGROUND ---
        # Stars
        for star in self.stars:
            screen_x = (star['pos'][0] - self.camera_x * star['depth']) % self.WORLD_WIDTH
            if 0 <= screen_x <= self.SCREEN_WIDTH:
                alpha = int(100 * star['depth'] + 50)
                color = (min(255, self.COLOR_BG[0]+alpha), min(255, self.COLOR_BG[1]+alpha), min(255, self.COLOR_BG[2]+alpha))
                pygame.draw.circle(self.screen, color, (int(screen_x), int(star['pos'][1])), star['size'])

        # Racetrack
        track_y = self.SCREEN_HEIGHT - 20
        track_y_top = 20
        self._draw_glow_line((0, track_y), (self.SCREEN_WIDTH, track_y), self.COLOR_TRACK_LINE, self.COLOR_TRACK_GLOW, 2, 8)
        self._draw_glow_line((0, track_y_top), (self.SCREEN_WIDTH, track_y_top), self.COLOR_TRACK_LINE, self.COLOR_TRACK_GLOW, 2, 8)

        # Finish Line
        finish_screen_x = self.FINISH_LINE_X - self.camera_x
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
            width = 10 + pulse * 10
            alpha = 150 + pulse * 105
            self._draw_glow_line((finish_screen_x, 0), (finish_screen_x, self.SCREEN_HEIGHT), self.COLOR_FINISH_LINE, self.COLOR_FINISH_GLOW, int(width), int(width*2), alpha)

        # --- ENTITIES ---
        # Particles (drawn first, behind other things)
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = int(255 * (p['lifetime'] / 20)) if p['lifetime'] < 20 else 255
            self._draw_glow_circle(pos, int(p['size']), p['color'], (p['color'][0]//2, p['color'][1]//2, p['color'][2]//2), alpha)

        # Towers
        for tower in self.towers:
            pos = (int(tower['pos'][0] - self.camera_x), int(tower['pos'][1]))
            self._draw_glow_circle(pos, 8, self.COLOR_TOWER, self.COLOR_TOWER_GLOW)

        # Creatures
        for creature in self.creatures:
            pos = (int(creature['pos'][0] - self.camera_x), int(creature['pos'][1]))
            self._draw_glow_circle(pos, int(creature['size']), self.COLOR_CREATURE, self.COLOR_CREATURE_GLOW)

        # Bullets
        for bullet in self.bullets:
            pos = (int(bullet['pos'][0] - self.camera_x), int(bullet['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_BULLET)

        # Drone
        drone_screen_pos = (int(self.drone_pos[0] - self.camera_x), int(self.drone_pos[1]))
        self._draw_glow_circle(drone_screen_pos, int(self.drone_size), self.COLOR_DRONE, self.COLOR_DRONE_GLOW)

        # Time Slow Vignette
        if self.is_time_slowing:
            vignette_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(70 * (self.time_slow_meter / self.TIME_SLOW_MAX))
            pygame.draw.rect(vignette_surface, self.COLOR_TIME_SLOW_VIGNETTE + (alpha,), (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.screen.blit(vignette_surface, (0, 0))

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Drone Size Bar
        size_percent = (self.drone_size - self.DRONE_MIN_SIZE) / (self.DRONE_START_SIZE - self.DRONE_MIN_SIZE)
        size_bar_width = 100
        pygame.draw.rect(self.screen, (50,50,50), (self.SCREEN_WIDTH - size_bar_width - 10, 10, size_bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_DRONE, (self.SCREEN_WIDTH - size_bar_width - 10, 10, size_bar_width * size_percent, 15))
        
        # Time Slow Gauge (around drone)
        drone_screen_pos = (int(self.drone_pos[0] - self.camera_x), int(self.drone_pos[1]))
        radius = int(self.drone_size + 8)
        if self.time_slow_meter > 0:
            angle = 360 * (self.time_slow_meter / self.TIME_SLOW_MAX)
            if angle > 0:
                rect = (drone_screen_pos[0] - radius, drone_screen_pos[1] - radius, radius*2, radius*2)
                pygame.draw.arc(self.screen, (220,220,255), rect, math.radians(90), math.radians(90 + angle), 2)

    def _draw_glow_circle(self, pos, radius, color, glow_color, alpha=255):
        if radius <= 0: return
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Glow
        pygame.gfxdraw.filled_circle(surf, radius, radius, radius, glow_color + (int(alpha*0.5),))
        
        # Main circle
        pygame.gfxdraw.filled_circle(surf, radius, radius, int(radius * 0.8), color + (alpha,))
        pygame.gfxdraw.aacircle(surf, radius, radius, int(radius * 0.8), color + (alpha,))
        
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_glow_line(self, p1, p2, color, glow_color, width, glow_width, alpha=255):
        # This is a simplified glow using thick lines.
        pygame.draw.line(self.screen, glow_color, p1, p2, glow_width)
        pygame.draw.line(self.screen, color, p1, p2, width)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "drone_pos_x": self.drone_pos[0],
            "distance_to_finish": self.FINISH_LINE_X - self.drone_pos[0],
        }

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It has been modified to use the correct API
    
    # Un-comment the following line to run with display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dreamscape Drone")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated and not truncated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        if keys[pygame.K_q]: terminated = True

        action = [movement, space_held, shift_held]
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"--- Game Over ---")
            print(f"Final Score: {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Total Steps: {info['steps']}")

    env.close()