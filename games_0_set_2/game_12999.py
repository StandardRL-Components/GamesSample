import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:46:45.161769
# Source Brief: brief_02999.md
# Brief Index: 2999
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race a futuristic carriage on a winding track, collecting crystals for boosts. "
        "Use your energy to create clones or reverse time to gain an edge."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to turn. "
        "Press space to create a clone and hold shift to reverse time."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_TRACK = (50, 50, 80)
    COLOR_TRACK_BORDER = (90, 90, 120)
    COLOR_FINISH_GOLD = (255, 215, 0)
    
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_CLONE = (100, 200, 255, 150)
    COLOR_AI = (255, 80, 50)
    
    COLOR_CRYSTAL = (200, 50, 255)
    COLOR_CRYSTAL_GLOW = (200, 50, 255, 60)
    COLOR_OBSTACLE = (255, 0, 0)
    COLOR_OBSTACLE_GLOW = (255, 0, 0, 60)
    
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_BAR = (0, 255, 200)
    COLOR_ENERGY_BAR_BG = (50, 50, 50)

    # Game Mechanics
    MAX_STEPS = 2500
    CARRIAGE_SIZE = 10
    TURN_SPEED = 4.5
    ACCELERATION = 0.2
    BRAKING = 0.3
    MAX_SPEED = 5.0
    FRICTION = 0.98
    BOOST_SPEED_BONUS = 3.0
    BOOST_DURATION = 90 # steps

    MAX_ENERGY = 100.0
    ENERGY_REGEN = 0.1
    CLONE_COST = 30.0
    TIME_REVERSE_COST = 0.5 # per step

    HISTORY_LENGTH = 300 # steps to remember for time travel/cloning

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables are initialized in reset()
        self.player = {}
        self.ai_opponent = {}
        self.clones = []
        self.obstacles = []
        self.crystals = []
        self.particles = []
        self.track_points = []
        self.finish_line = None
        self.player_history = deque(maxlen=self.HISTORY_LENGTH)
        
        self.last_space_held = False
        self.time_reversing = False
        self.races_won = 0

        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self._generate_track_and_objects()

        self.player = {
            "pos": self.track_points[0].copy() + pygame.Vector2(-20, 0),
            "vel": pygame.Vector2(0, 0),
            "angle": 0,
            "energy": self.MAX_ENERGY,
            "boost_timer": 0,
            "progress": 0,
            "rank": 2
        }
        
        self.ai_opponent = {
            "pos": self.track_points[0].copy() + pygame.Vector2(20, 0),
            "vel": pygame.Vector2(0, 0),
            "angle": 0,
            "target_point_idx": 1,
            "speed": self.MAX_SPEED * 0.85,
            "progress": 0,
            "rank": 1
        }
        
        self.clones = []
        self.particles = []
        self.player_history.clear()
        
        self.last_space_held = False
        self.time_reversing = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step

        # --- Handle Abilities ---
        self._handle_time_reversal(shift_held)
        self._handle_cloning(space_held)

        # --- Update Game Logic ---
        if not self.time_reversing:
            self._update_player(movement)
        
        self._update_ai()
        self._update_clones()
        self._update_obstacles()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        reward += self._handle_collisions()
        
        # --- Update Game State ---
        self.steps += 1
        self.player['energy'] = min(self.MAX_ENERGY, self.player['energy'] + self.ENERGY_REGEN)
        if self.player['boost_timer'] > 0:
            self.player['boost_timer'] -= 1

        # --- Update Ranks ---
        self._update_ranks()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limit
        if terminated:
            if self.win_condition:
                reward += 100
                self.races_won += 1
            else:
                reward -= 100
            self.game_over = True

        self.last_space_held = space_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_track_and_objects(self):
        self.track_points = []
        center_y = self.HEIGHT / 2
        amplitude = self.HEIGHT / 3
        frequency = 0.01
        for i in range(self.WIDTH + 200):
            x = i * 4
            y = center_y + math.sin(x * frequency + self.races_won * 0.5) * amplitude
            if x < self.WIDTH + 100:
                self.track_points.append(pygame.Vector2(x, y))

        finish_x = self.track_points[-20].x
        self.finish_line = pygame.Rect(finish_x, 0, 10, self.HEIGHT)
        
        self.crystals = []
        for i in range(10, len(self.track_points) - 30, 15):
             if self.np_random.random() < 0.5:
                offset = pygame.Vector2(0, self.np_random.uniform(-40, 40)).rotate(-math.degrees(self.track_points[i].angle_to(self.track_points[i+1]-self.track_points[i])))
                self.crystals.append(self.track_points[i] + offset)

        self.obstacles = []
        obstacle_speed = 1.0 + self.races_won * 0.05
        for i in range(20, len(self.track_points) - 40, 25):
            if self.np_random.random() < 0.4:
                pos = self.track_points[i].copy()
                vel = pygame.Vector2(self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])) * obstacle_speed
                self.obstacles.append({"pos": pos, "vel": vel, "size": 8})

    def _handle_time_reversal(self, shift_held):
        if shift_held and self.player['energy'] > self.TIME_REVERSE_COST and len(self.player_history) > 1:
            self.time_reversing = True
            self.player['energy'] -= self.TIME_REVERSE_COST
            last_state = self.player_history.pop()
            self.player['pos'], self.player['angle'] = last_state['pos'], last_state['angle']
            self.player['vel'] *= 0.5 # Dampen velocity to avoid jerky movement
            # Sound: Time reverse whoosh
        else:
            self.time_reversing = False

    def _handle_cloning(self, space_held):
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.player['energy'] >= self.CLONE_COST:
            self.player['energy'] -= self.CLONE_COST
            clone = {
                "path": list(self.player_history),
                "path_idx": 0,
                "pos": self.player['pos'].copy(),
                "angle": self.player['angle']
            }
            self.clones.append(clone)
            self._create_particles(self.player['pos'], 20, (255, 255, 255), 2.0, 30)
            # Sound: Clone activate

    def _update_player(self, movement):
        # Turning
        if movement == 3: # Left
            self.player['angle'] -= self.TURN_SPEED
        if movement == 4: # Right
            self.player['angle'] += self.TURN_SPEED
        
        # Acceleration
        current_max_speed = self.MAX_SPEED + (self.BOOST_SPEED_BONUS if self.player['boost_timer'] > 0 else 0)
        if movement == 1: # Up/Accelerate
            acceleration_vector = pygame.Vector2(1, 0).rotate(self.player['angle']) * self.ACCELERATION
            self.player['vel'] += acceleration_vector
        elif movement == 2: # Down/Brake
            self.player['vel'] *= (1.0 - self.BRAKING)
        
        # Physics
        self.player['vel'] *= self.FRICTION
        if self.player['vel'].length() > current_max_speed:
            self.player['vel'].scale_to_length(current_max_speed)
        
        self.player['pos'] += self.player['vel']
        
        # History
        self.player_history.append({'pos': self.player['pos'].copy(), 'angle': self.player['angle']})
        
        # Boost particles
        if self.player['boost_timer'] > 0:
            self._create_particles(self.player['pos'], 2, self.COLOR_PLAYER, 1.5, 20, -self.player['angle'])

    def _update_ai(self):
        if len(self.track_points) <= self.ai_opponent['target_point_idx']:
            return
        target_pos = self.track_points[self.ai_opponent['target_point_idx']]
        direction_vec = target_pos - self.ai_opponent['pos']
        
        if direction_vec.length() < 20:
            self.ai_opponent['target_point_idx'] = min(len(self.track_points) - 1, self.ai_opponent['target_point_idx'] + 1)
        
        self.ai_opponent['angle'] = direction_vec.angle_to(pygame.Vector2(1, 0))
        self.ai_opponent['pos'] += direction_vec.normalize() * self.ai_opponent['speed']

    def _update_clones(self):
        for clone in self.clones:
            if clone['path_idx'] < len(clone['path']):
                state = clone['path'][clone['path_idx']]
                clone['pos'], clone['angle'] = state['pos'], state['angle']
                clone['path_idx'] += 1
            # Optional: remove finished clones
            # else: self.clones.remove(clone)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'] += obs['vel']
            if obs['pos'].x < 0 or obs['pos'].x > self.WIDTH: obs['vel'].x *= -1
            if obs['pos'].y < 0 or obs['pos'].y > self.HEIGHT: obs['vel'].y *= -1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player vs Crystals
        for crystal_pos in self.crystals[:]:
            if self.player['pos'].distance_to(crystal_pos) < self.CARRIAGE_SIZE + 5:
                self.crystals.remove(crystal_pos)
                self.player['boost_timer'] = self.BOOST_DURATION
                reward += 0.5
                # Sound: Crystal collect
        
        # Player vs Obstacles
        for obs in self.obstacles:
            if self.player['pos'].distance_to(obs['pos']) < self.CARRIAGE_SIZE + obs['size']:
                self.game_over = True
                # Sound: Crash
        
        # Player vs Track Boundaries
        min_dist_sq = float('inf')
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            l2 = p1.distance_squared_to(p2)
            if l2 == 0: continue
            t = max(0, min(1, (self.player['pos'] - p1).dot(p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            min_dist_sq = min(min_dist_sq, self.player['pos'].distance_squared_to(projection))
        
        if math.sqrt(min_dist_sq) > 50: # Track width is 100
             self.game_over = True
             # Sound: Fall off track
        
        return reward

    def _update_ranks(self):
        # Simple progress based on x-coordinate
        self.player['progress'] = self.player['pos'].x
        self.ai_opponent['progress'] = self.ai_opponent['pos'].x
        
        last_rank = self.player['rank']
        if self.player['progress'] > self.ai_opponent['progress']:
            self.player['rank'] = 1
            self.ai_opponent['rank'] = 2
        else:
            self.player['rank'] = 2
            self.ai_opponent['rank'] = 1
            
        if self.player['rank'] < last_rank:
            self.score += 1 # Reward for overtaking

    def _check_termination(self):
        if self.player['pos'].x >= self.finish_line.x:
            self.win_condition = True
            return True
        if self.ai_opponent['pos'].x >= self.finish_line.x:
            self.win_condition = False
            return True
        if self.game_over: # Crashed
            self.win_condition = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.win_condition = self.player['progress'] > self.ai_opponent['progress']
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Track
        pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, False, self.track_points, 100)
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, self.track_points, 90)

        # Finish Line
        pygame.draw.rect(self.screen, self.COLOR_FINISH_GOLD, self.finish_line)
        for i in range(0, self.HEIGHT, 20):
             pygame.draw.rect(self.screen, self.COLOR_BG, (self.finish_line.x, i, 5, 10))
             pygame.draw.rect(self.screen, self.COLOR_BG, (self.finish_line.x+5, i+10, 5, 10))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            p_color = p['color'] + (alpha,) if len(p['color']) == 3 else p['color']
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                self._draw_circle(p['pos'], radius, p_color)

        # Crystals
        for pos in self.crystals:
            self._draw_glowing_polygon(pos, 6, 8, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_GLOW, self.steps * 2)

        # Obstacles
        for obs in self.obstacles:
            self._draw_circle(obs['pos'], obs['size'] + 4, self.COLOR_OBSTACLE_GLOW)
            self._draw_circle(obs['pos'], obs['size'], self.COLOR_OBSTACLE)

        # Clones
        for clone in self.clones:
            self._draw_carriage(clone['pos'], clone['angle'], self.COLOR_CLONE, self.CARRIAGE_SIZE)

        # AI Opponent
        self._draw_carriage(self.ai_opponent['pos'], self.ai_opponent['angle'], self.COLOR_AI, self.CARRIAGE_SIZE, True)

        # Player
        self._draw_carriage(self.player['pos'], self.player['angle'], self.COLOR_PLAYER, self.CARRIAGE_SIZE, True)

        # Time Reversal Effect
        if self.time_reversing:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((100, 200, 255, 30))
            for i in range(10):
                y = self.np_random.integers(0, self.HEIGHT)
                pygame.draw.line(overlay, (255, 255, 255, 50), (0, y), (self.WIDTH, y), 1)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Race Position
        pos_text = self.font_large.render(f"{self.player['rank']}/2", True, self.COLOR_UI_TEXT)
        self.screen.blit(pos_text, (20, 10))

        # Lap Time
        time_seconds = self.steps / self.FPS
        time_text = self.font_large.render(f"{time_seconds:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))
        
        # Energy Bar
        bar_width, bar_height = 200, 20
        bar_x, bar_y = self.WIDTH / 2 - bar_width / 2, self.HEIGHT - 40
        energy_ratio = self.player['energy'] / self.MAX_ENERGY
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, bar_width * energy_ratio, bar_height), border_radius=5)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "rank": self.player['rank']}

    def _create_particles(self, pos, count, color, speed, life, angle_offset=None):
        for _ in range(count):
            if angle_offset is not None:
                angle = math.radians(angle_offset + self.np_random.uniform(-30, 30))
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
            
            p_speed = self.np_random.uniform(speed * 0.5, speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * p_speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": life, "max_life": life, 
                "color": color, "size": self.np_random.uniform(2, 5)
            })

    def _draw_carriage(self, pos, angle, color, size, has_glow=False):
        if has_glow:
            glow_color = color + (50,) if len(color) == 3 else color[:3] + (50,)
            self._draw_circle(pos, size * 2.5, glow_color)

        points = [
            pygame.Vector2(size, 0),
            pygame.Vector2(-size * 0.7, size * 0.8),
            pygame.Vector2(-size * 0.5, 0),
            pygame.Vector2(-size * 0.7, -size * 0.8),
        ]
        rotated_points = [p.rotate(angle) + pos for p in points]
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, color)

    def _draw_glowing_polygon(self, pos, num_sides, radius, color, glow_color, rotation=0):
        points = []
        for i in range(num_sides):
            angle = (2 * math.pi / num_sides) * i + math.radians(rotation)
            p = pygame.Vector2(math.cos(angle), math.sin(angle)) * radius + pos
            points.append((int(p.x), int(p.y)))
        
        # Glow
        self._draw_circle(pos, radius * 2.5, glow_color)
        
        # Shape
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_circle(self, pos, radius, color):
        x, y = int(pos.x), int(pos.y)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)

    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Crystal Carriage Racer")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        movement = 0 # None
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()