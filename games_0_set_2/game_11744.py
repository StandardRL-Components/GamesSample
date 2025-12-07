import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:28:17.326398
# Source Brief: brief_01744.md
# Brief Index: 1744
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a transforming particle swarm
    through a windy field of obstacles to reach a finish line. The swarm's shape,
    controlled by the player, determines its interaction with the wind.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a transforming particle swarm through a windy field of obstacles. "
        "Change the swarm's shape to manipulate its interaction with the wind and reach the finish line."
    )
    user_guide = (
        "Controls: Hold space to make the swarm compact and less affected by wind. "
        "Hold shift to spread the swarm out, making it more susceptible to wind."
    )
    auto_advance = True


    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 120 * FPS  # 120 seconds

    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_PARTICLE = (100, 180, 255)
    COLOR_GLOW = (50, 90, 150)
    COLOR_OBSTACLE = (220, 50, 50)
    COLOR_FINISH_LINE = (50, 220, 50)
    COLOR_WIND = (40, 50, 80)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_BG = (25, 30, 55, 180) # RGBA

    # Game Parameters
    INITIAL_PARTICLES = 50
    FINISH_LINE_X = 620
    WIND_CHANGE_INTERVAL = 10 * FPS # 10 seconds
    MIN_WIND_SPEED = 0.3
    MAX_WIND_SPEED = 0.8

    # Shape Drag Modifiers
    DRAG_MODIFIERS = {
        'neutral': 0.8,
        'compact': 0.3,
        'spread': 1.5
    }

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # State variables will be initialized in reset()
        self.swarms = []
        self.obstacles = []
        self.wind_lines = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0
        self.wind_vector = pygame.Vector2(0, 0)
        self.wind_timer = 0
        self.last_max_x = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS

        # --- Initialize Swarm ---
        self.swarms = [self._create_swarm(self.INITIAL_PARTICLES, pygame.Vector2(50, self.HEIGHT / 2))]
        self.last_max_x = max(s['center'].x for s in self.swarms) if self.swarms else 0

        # --- Initialize Obstacles ---
        self.obstacles = [self._create_obstacle() for _ in range(8)]

        # --- Initialize Wind ---
        self._update_wind(force_update=True)
        self.wind_lines = [self._create_wind_line() for _ in range(50)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        _, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_actions(space_held, shift_held)

        # --- Game Logic ---
        self.steps += 1
        self.timer -= 1
        
        self._update_wind()
        self._update_wind_lines()
        self._update_obstacles()
        self._update_swarms()
        self._check_collisions()
        merge_reward = self._check_merging()

        # --- Reward Calculation ---
        reward = self._calculate_reward(merge_reward)
        self.score += reward

        # --- Termination Check ---
        terminated = self._check_termination()
        self.game_over = terminated
        truncated = False # This environment does not truncate based on steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_swarm(self, num_particles, center):
        swarm = {
            'center': pygame.Vector2(center),
            'velocity': pygame.Vector2(0, 0),
            'shape': 'neutral',
            'particles': [],
            'radius': math.sqrt(num_particles) * 5
        }
        for _ in range(num_particles):
            swarm['particles'].append({
                'pos': pygame.Vector2(center),
                'offset': pygame.Vector2(0, 0)
            })
        self._update_particle_targets(swarm)
        return swarm

    def _create_obstacle(self):
        return pygame.Rect(
            random.uniform(150, self.WIDTH - 50),
            random.uniform(0, self.HEIGHT - 50),
            random.uniform(15, 40),
            random.uniform(40, 100)
        )

    def _create_wind_line(self, on_screen=True):
        if on_screen:
            x = random.uniform(0, self.WIDTH)
            y = random.uniform(0, self.HEIGHT)
        else: # Respawn off-screen
            if self.wind_vector.length() > 0 and abs(self.wind_vector.x) > abs(self.wind_vector.y):
                x = -20 if self.wind_vector.x > 0 else self.WIDTH + 20
                y = random.uniform(0, self.HEIGHT)
            else:
                x = random.uniform(0, self.WIDTH)
                y = -20 if self.wind_vector.y > 0 else self.HEIGHT + 20
        return pygame.Vector2(x, y)

    def _handle_actions(self, space_held, shift_held):
        # space_held is prioritized over shift_held
        new_shape = 'neutral'
        if space_held:
            new_shape = 'compact'
        elif shift_held:
            new_shape = 'spread'

        for swarm in self.swarms:
            if swarm['shape'] != new_shape:
                swarm['shape'] = new_shape
                self._update_particle_targets(swarm)
                # SFX: Swarm transform sound

    def _update_wind(self, force_update=False):
        self.wind_timer -= 1
        if self.wind_timer <= 0 or force_update:
            self.wind_timer = self.WIND_CHANGE_INTERVAL
            angle = random.uniform(-45, 45)
            speed = random.uniform(self.MIN_WIND_SPEED, self.MAX_WIND_SPEED)
            self.wind_vector = pygame.Vector2(speed, 0).rotate(angle)
            # SFX: Wind gust sound

    def _update_wind_lines(self):
        for i, pos in enumerate(self.wind_lines):
            pos += self.wind_vector * 5 # Visual speed
            if not (-20 < pos.x < self.WIDTH + 20 and -20 < pos.y < self.HEIGHT + 20):
                self.wind_lines[i] = self._create_wind_line(on_screen=False)

    def _update_particle_targets(self, swarm):
        num_particles = len(swarm['particles'])
        if num_particles == 0: return

        radius = math.sqrt(num_particles) * 5
        swarm['radius'] = radius

        for i, p in enumerate(swarm['particles']):
            angle = (i / num_particles) * 2 * math.pi if num_particles > 0 else 0
            if swarm['shape'] == 'compact':
                r = random.uniform(0, radius * 0.5)
            elif swarm['shape'] == 'neutral':
                r = random.uniform(0, radius)
            elif swarm['shape'] == 'spread':
                r = random.uniform(0, radius)
                p['offset'].x = math.cos(angle) * r * 0.5
                p['offset'].y = math.sin(angle) * r * 1.5
                if self.wind_vector.length() > 0:
                    p['offset'].rotate_ip(-self.wind_vector.angle_to(pygame.Vector2(1,0)))
                continue
            
            p['offset'].x = math.cos(angle) * r
            p['offset'].y = math.sin(angle) * r

    def _update_swarms(self):
        for swarm in self.swarms:
            if not swarm['particles']: continue

            drag = self.DRAG_MODIFIERS[swarm['shape']]
            force = self.wind_vector * drag
            
            swarm['velocity'] += force
            swarm['velocity'] *= 0.95 # Damping
            swarm['center'] += swarm['velocity']

            swarm['center'].x = max(0, min(self.WIDTH, swarm['center'].x))
            swarm['center'].y = max(0, min(self.HEIGHT, swarm['center'].y))

            for p in swarm['particles']:
                target_pos = swarm['center'] + p['offset']
                target_pos += pygame.Vector2(random.uniform(-1,1), random.uniform(-1,1))
                p['pos'] = p['pos'].lerp(target_pos, 0.2)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs.x += self.wind_vector.x * 0.5
            obs.y += self.wind_vector.y * 0.5

            if obs.right < 0 or obs.left > self.WIDTH or obs.bottom < 0 or obs.top > self.HEIGHT:
                side = random.choice(['left', 'right', 'top', 'bottom'])
                if side == 'left':
                    obs.right = 0; obs.y = random.uniform(0, self.HEIGHT - obs.height)
                elif side == 'right':
                    obs.left = self.WIDTH; obs.y = random.uniform(0, self.HEIGHT - obs.height)
                elif side == 'top':
                    obs.bottom = 0; obs.x = random.uniform(0, self.WIDTH - obs.width)
                else: # bottom
                    obs.top = self.HEIGHT; obs.x = random.uniform(0, self.WIDTH - obs.width)

    def _check_collisions(self):
        swarms_to_add = []
        swarms_to_remove = []

        for swarm in self.swarms:
            if not swarm['particles']: continue
            
            for obs in self.obstacles:
                closest_x = max(obs.left, min(swarm['center'].x, obs.right))
                closest_y = max(obs.top, min(swarm['center'].y, obs.bottom))
                distance = swarm['center'].distance_to(pygame.Vector2(closest_x, closest_y))

                if distance < swarm['radius'] and len(swarm['particles']) > 1:
                    # SFX: Swarm split sound
                    swarms_to_remove.append(swarm)
                    
                    p1_count = len(swarm['particles'])//2
                    
                    collision_normal = (swarm['center'] - pygame.Vector2(obs.center)).normalize()
                    if collision_normal.length() == 0: collision_normal = pygame.Vector2(1,0)

                    c1 = swarm['center'] + collision_normal * 15
                    c2 = swarm['center'] - collision_normal * 15

                    swarm1 = self._create_swarm(p1_count, c1)
                    swarm1['velocity'] = swarm['velocity'] + collision_normal * 1.5
                    
                    swarm2 = self._create_swarm(len(swarm['particles']) - p1_count, c2)
                    swarm2['velocity'] = swarm['velocity'] - collision_normal * 1.5

                    swarms_to_add.extend([swarm1, swarm2])
                    break
        
        if swarms_to_remove:
            self.swarms = [s for s in self.swarms if s not in swarms_to_remove]
            self.swarms.extend(swarms_to_add)

    def _check_merging(self):
        if len(self.swarms) < 2: return 0
        
        merged_swarms = set()
        new_swarms = []
        reward = 0

        for i in range(len(self.swarms)):
            for j in range(i + 1, len(self.swarms)):
                s1, s2 = self.swarms[i], self.swarms[j]
                if i in merged_swarms or j in merged_swarms or not s1['particles'] or not s2['particles']: continue

                dist = s1['center'].distance_to(s2['center'])
                if dist < s1['radius'] + s2['radius']:
                    # SFX: Swarm merge sound
                    merged_swarms.add(i); merged_swarms.add(j)
                    
                    total_particles = len(s1['particles']) + len(s2['particles'])
                    w1 = len(s1['particles']) / total_particles if total_particles > 0 else 0.5
                    w2 = 1.0 - w1

                    new_center = s1['center'].lerp(s2['center'], w2)
                    new_swarm = self._create_swarm(total_particles, new_center)
                    new_swarm['velocity'] = s1['velocity'].lerp(s2['velocity'], w2)
                    new_swarms.append(new_swarm)
                    reward += 1.0

        if merged_swarms:
            self.swarms = [self.swarms[i] for i in range(len(self.swarms)) if i not in merged_swarms] + new_swarms
        
        return reward

    def _calculate_reward(self, merge_reward):
        reward = -0.01 # Time penalty
        
        if self.swarms:
            current_max_x = max(s['center'].x for s in self.swarms if s['particles'])
            progress = current_max_x - self.last_max_x
            reward += progress * 0.1
            self.last_max_x = current_max_x
        
        reward += merge_reward
        
        if any(s['center'].x + s['radius'] > self.FINISH_LINE_X for s in self.swarms if s['particles']):
            reward += 100
        elif self.timer <= 0:
            reward -= 100
            
        return reward

    def _check_termination(self):
        win = any(s['center'].x + s['radius'] > self.FINISH_LINE_X for s in self.swarms if s['particles'])
        timeout = self.timer <= 0
        return win or timeout

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "num_swarms": len(self.swarms),
            "total_particles": sum(len(s['particles']) for s in self.swarms)
        }

    def _render_game(self):
        for pos in self.wind_lines:
            end_pos = pos - self.wind_vector.normalize() * 5 if self.wind_vector.length() > 0 else pos
            pygame.draw.line(self.screen, self.COLOR_WIND, (int(pos.x), int(pos.y)), (int(end_pos.x), int(end_pos.y)), 1)
        
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.HEIGHT), 3)

        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

        for swarm in self.swarms:
            for p in swarm['particles']:
                glow_radius = 8
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_GLOW, 80), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (int(p['pos'].x - glow_radius), int(p['pos'].y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, self.COLOR_PARTICLE)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, self.COLOR_PARTICLE)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        seconds_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {seconds_left:.1f}"
        timer_render = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_render, (self.WIDTH - timer_render.get_width() - 10, 8))

        if self.wind_vector.length() > 0:
            wind_angle = self.wind_vector.angle_to(pygame.Vector2(1, 0))
            center, arrow_len = pygame.Vector2(30, 20), 15
            end = center + pygame.Vector2(arrow_len, 0).rotate(-wind_angle)
            p1 = end + pygame.Vector2(-7, -4).rotate(-wind_angle)
            p2 = end + pygame.Vector2(-7, 4).rotate(-wind_angle)
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, center, end, 2)
            pygame.draw.polygon(self.screen, self.COLOR_UI_TEXT, [end, p1, p2])

        num_swarms = len(self.swarms)
        total_particles = sum(len(s['particles']) for s in self.swarms)
        swarm_info_text = f"SWARMS: {num_swarms} | PARTICLES: {total_particles}"
        swarm_info_render = self.font_small.render(swarm_info_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(swarm_info_render, (10, self.HEIGHT - swarm_info_render.get_height() - 5))
        
        score_text = f"SCORE: {self.score:.1f}"
        score_render = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_render, (self.WIDTH // 2 - score_render.get_width() // 2, 8))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("--- Running Implementation Validation ---")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2], "Action space incorrect"
        print("✓ Action space is correct.")
        
        obs, info = self.reset()
        assert isinstance(obs, np.ndarray), f"Observation is {type(obs)}"
        assert obs.shape == self.observation_space.shape, f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8, f"Obs dtype is {obs.dtype}"
        assert isinstance(info, dict), f"Info is {type(info)}"
        print("✓ reset() returns correct format.")
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert isinstance(obs, np.ndarray) and obs.shape == self.observation_space.shape, "Step obs incorrect"
        assert isinstance(reward, (int, float)), f"Reward is {type(reward)}"
        assert isinstance(term, bool), f"Terminated is {type(term)}"
        assert not trunc, f"Truncated is not False"
        assert isinstance(info, dict), f"Info is {type(info)}"
        print("✓ step() returns correct format.")
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    # Un-comment the following line to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    # The main loop is for demonstration and debugging.
    # It requires a display to run.
    if "SDL_VIDEODRIVER" not in os.environ:
        pygame.display.set_caption("Particle Swarm Environment")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0.0
        
        while running:
            keys = pygame.key.get_pressed()
            # Action format: [no-op, space, shift]
            action = [0, 1 if keys[pygame.K_SPACE] else 0, 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0.0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0.0

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)
            
        env.close()
    else:
        print("\nRunning in headless mode. No display will be shown.")
        print("To run with a display, comment out the `os.environ.setdefault` line at the top of the file.")