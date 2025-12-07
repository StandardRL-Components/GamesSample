
# Generated: 2025-08-28T03:54:57.176920
# Source Brief: brief_02157.md
# Brief Index: 2157

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # The provided user guide is for a top-down racer. This one is adapted for the side-scrolling
    # concept, interpreting "turn" as "change lanes".
    user_guide = (
        "Controls: ←→ to change lanes. Hold Space for a speed boost."
    )

    game_description = (
        "A retro-futuristic racer. Dodge red obstacles, collect gold boosts, and complete 3 laps to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.W, self.H = 640, 400
        self.FPS = 30
        self.LAPS_TO_WIN = 3
        self.MAX_STEPS = 5000  # Increased to allow for 3 laps

        # --- Colors ---
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_STARS = (50, 40, 80)
        self.COLOR_TRACK = (40, 20, 80)
        self.COLOR_TRACK_LINES = (70, 50, 130)
        self.COLOR_CAR = (220, 220, 255)
        self.COLOR_COCKPIT = (50, 50, 80)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_BOOST = (255, 220, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game Properties ---
        self.TRACK_Y = self.H * 0.3
        self.TRACK_H = self.H * 0.6
        self.NUM_LANES = 4
        self.LANE_H = self.TRACK_H / self.NUM_LANES
        self.SEGMENT_W = 200
        self.LAP_LENGTH = 15000  # pixels

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.laps_completed = 0
        self.lap_start_step = 0
        self.car_pos = None
        self.car_lane = 0
        self.car_target_y = 0
        self.car_tilt = 0.0
        self.scroll_x = 0.0
        self.base_scroll_speed = 10.0
        self.scroll_speed = 0.0
        self.boost_timer = 0
        self.obstacles = deque()
        self.boosts = deque()
        self.particles = deque()
        self.stars = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.laps_completed = 0
        self.lap_start_step = 0

        # --- Reset Car State ---
        self.car_lane = self.NUM_LANES // 2
        self.car_pos = pygame.Vector2(self.W * 0.2, self._get_y_for_lane(self.car_lane))
        self.car_target_y = self.car_pos.y
        self.car_tilt = 0.0

        # --- Reset Track State ---
        self.scroll_x = 0.0
        self.base_scroll_speed = 10.0
        self.scroll_speed = self.base_scroll_speed
        self.boost_timer = 0

        # --- Reset Entities ---
        self.obstacles.clear()
        self.boosts.clear()
        self.particles.clear()
        self._generate_track(full_regen=True)
        self._generate_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0.1  # Survival reward

        self._handle_input(action)
        self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        reward = self.reward_this_step
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Map left/right actions to changing lanes up/down
        target_lane = self.car_lane
        if movement == 3:  # Left -> Move Up
            target_lane -= 1
            self.car_tilt = -15
        elif movement == 4:  # Right -> Move Down
            target_lane += 1
            self.car_tilt = 15
        
        self.car_lane = np.clip(target_lane, 0, self.NUM_LANES - 1)
        self.car_target_y = self._get_y_for_lane(self.car_lane)

        if space_held and self.boost_timer <= 0:
            self.boost_timer = self.FPS * 2  # 2-second boost
            # Sound: Boost activate
            for _ in range(20):
                self._create_particle(self.car_pos, self.COLOR_BOOST, speed_mult=2.0)

    def _update_game_state(self):
        # --- Update Boost and Speed ---
        if self.boost_timer > 0:
            self.boost_timer -= 1
            self.scroll_speed = self.base_scroll_speed * 2.0
            if self.steps % 2 == 0:
                p_pos = self.car_pos - (20, 0)
                self._create_particle(p_pos, self.COLOR_BOOST, speed_mult=0.5, life=10)
        else:
            self.scroll_speed = self.base_scroll_speed

        # --- Update Scroll Position ---
        self.scroll_x += self.scroll_speed

        # --- Update Car Position and Animation ---
        self.car_pos.y += (self.car_target_y - self.car_pos.y) * 0.25
        self.car_tilt *= 0.9

        # --- Update Lap Counter ---
        if self.scroll_x // self.LAP_LENGTH > self.laps_completed:
            self.laps_completed += 1
            self.lap_start_step = self.steps
            self.base_scroll_speed *= 1.05  # Increase difficulty
            if self.laps_completed >= self.LAPS_TO_WIN:
                self.reward_this_step += 500  # Game win reward
            else:
                self.reward_this_step += 100  # Lap completion reward
            # Sound: Lap complete

        # --- Update Entities ---
        self._generate_track()
        self._update_particles()
        self._check_collisions()

    def _check_collisions(self):
        car_rect = pygame.Rect(self.car_pos.x - 18, self.car_pos.y - 8, 36, 16)

        # Obstacle collision
        for obs in self.obstacles:
            if car_rect.colliderect(obs):
                self.reward_this_step -= 50
                self.game_over = True
                # Sound: Crash
                for _ in range(50):
                    self._create_particle(self.car_pos, self.COLOR_OBSTACLE, speed_mult=3.0, life=40)
                return

        # Boost collection
        collected_boosts = []
        for boost in self.boosts:
            if car_rect.colliderect(boost):
                self.reward_this_step += 5
                self.boost_timer = self.FPS * 2
                collected_boosts.append(boost)
                # Sound: Boost collect
                for _ in range(20):
                    self._create_particle(self.car_pos, self.COLOR_BOOST, speed_mult=2.0)
        if collected_boosts:
            self.boosts = deque([b for b in self.boosts if b not in collected_boosts])

    def _generate_track(self, full_regen=False):
        if full_regen:
            self.obstacles.clear()
            self.boosts.clear()
            start_x = -self.SEGMENT_W
            end_x = self.W + self.SEGMENT_W
        else:
            # Remove off-screen entities
            self.obstacles = deque([o for o in self.obstacles if o.right > self.scroll_x])
            self.boosts = deque([b for b in self.boosts if b.right > self.scroll_x])
            # Add new segments if needed
            last_entity_x = 0
            if self.obstacles: last_entity_x = max(last_entity_x, self.obstacles[-1].x)
            if self.boosts: last_entity_x = max(last_entity_x, self.boosts[-1].x)
            start_x = last_entity_x + self.SEGMENT_W
            end_x = self.scroll_x + self.W + self.SEGMENT_W

        for x_pos in range(int(start_x), int(end_x), self.SEGMENT_W):
            obstacle_density = 0.2 + self.laps_completed * 0.05
            available_lanes = list(range(self.NUM_LANES))
            self.np_random.shuffle(available_lanes)
            
            num_obstacles_in_segment = self.np_random.integers(0, self.NUM_LANES)
            obstacle_lanes = available_lanes[:num_obstacles_in_segment]
            
            for lane_idx in range(self.NUM_LANES):
                if lane_idx in obstacle_lanes and self.np_random.random() < obstacle_density:
                    obs_x = x_pos + self.np_random.random() * self.SEGMENT_W * 0.8
                    obs_y = self.TRACK_Y + lane_idx * self.LANE_H
                    self.obstacles.append(pygame.Rect(obs_x, obs_y, self.SEGMENT_W * 0.2, self.LANE_H))
                elif lane_idx not in obstacle_lanes and self.np_random.random() < 0.1:
                    boost_x = x_pos + self.np_random.random() * self.SEGMENT_W
                    boost_y = self._get_y_for_lane(lane_idx)
                    self.boosts.append(pygame.Rect(boost_x - 10, boost_y - 10, 20, 20))

    def _get_y_for_lane(self, lane):
        return self.TRACK_Y + (lane + 0.5) * self.LANE_H

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.laps_completed >= self.LAPS_TO_WIN

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Track ---
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y, self.W, self.TRACK_H))
        for i in range(1, self.NUM_LANES + 1):
            y = self.TRACK_Y + i * self.LANE_H
            for x_start in range(0, self.W + 50, 50):
                line_x = x_start - (self.scroll_x * 1.5 % 50)
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (line_x, y), (line_x + 25, y), 2)

        # --- Draw Entities ---
        for obs in self.obstacles:
            screen_rect = obs.move(-self.scroll_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, tuple(c * 0.7 for c in self.COLOR_OBSTACLE), screen_rect, 2)
        for boost in self.boosts:
            pos = (int(boost.centerx - self.scroll_x), int(boost.centery))
            if -20 < pos[0] < self.W + 20:
                self._draw_glow_circle(self.screen, pos, self.COLOR_BOOST, 10, 20)

        # --- Draw Particles and Car ---
        self._render_particles()
        if not self.game_over:
            self._render_car()

    def _render_car(self):
        x, y = self.car_pos
        angle = math.radians(self.car_tilt)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        def rotate(p):
            px, py = p
            return (px * cos_a - py * sin_a + x, px * sin_a + py * cos_a + y)
        
        car_points = [rotate(p) for p in [(-18, 0), (10, -8), (18, 0), (10, 8)]]
        pygame.gfxdraw.aapolygon(self.screen, car_points, self.COLOR_CAR)
        pygame.gfxdraw.filled_polygon(self.screen, car_points, self.COLOR_CAR)
        
        c_pos = rotate((5, 0))
        pygame.gfxdraw.aacircle(self.screen, int(c_pos[0]), int(c_pos[1]), 4, self.COLOR_COCKPIT)
        pygame.gfxdraw.filled_circle(self.screen, int(c_pos[0]), int(c_pos[1]), 4, self.COLOR_COCKPIT)

    def _render_ui(self):
        lap_text = f"LAP: {min(self.laps_completed + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}"
        self.screen.blit(self.font_small.render(lap_text, True, self.COLOR_TEXT), (10, 10))
        
        time_elapsed = (self.steps - self.lap_start_step) / self.FPS
        time_text = f"TIME: {time_elapsed:.2f}"
        self.screen.blit(self.font_small.render(time_text, True, self.COLOR_TEXT), (10, 30))
        
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, score_surf.get_rect(topright=(self.W - 10, 10)))

        if self.game_over:
            msg = "FINISH!" if self.laps_completed >= self.LAPS_TO_WIN else "GAME OVER"
            end_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_surf, end_surf.get_rect(center=(self.W / 2, self.H / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "laps": self.laps_completed}

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append(
                (
                    self.np_random.random() * self.W,
                    self.np_random.random() * self.H,
                    self.np_random.random() * 1.5 + 0.5,
                )
            )

    def _render_stars(self):
        for x, y, speed in self.stars:
            screen_x = (x - self.scroll_x * speed * 0.1) % self.W
            pygame.gfxdraw.pixel(self.screen, int(screen_x), int(y), self.COLOR_STARS)

    def _draw_glow_circle(self, surface, pos, color, radius, glow_radius):
        glow_color = (*color, 50)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _create_particle(self, pos, color, speed_mult=1.0, life=20, size=3):
        angle = self.np_random.random() * math.pi * 2
        speed = self.np_random.random() * 2 * speed_mult
        vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'size': size})

    def _update_particles(self):
        self.particles = deque([p for p in self.particles if p['life'] > 0])

    def _render_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                alpha = max(0, int(255 * (p['life'] / p['max_life'])))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, p['pos'] - (p['size'], p['size']))

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")