
# Generated: 2025-08-28T02:03:23.674926
# Source Brief: brief_04316.md
# Brief Index: 4316

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the drawing cursor. Hold Space to move the cursor without drawing."
    )

    game_description = (
        "Draw a track for your sledder to ride on. Navigate the terrain, collect all checkpoints, and reach the finish line before time runs out!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.25
        self.RIDER_RADIUS = 7
        self.CURSOR_SPEED = 6
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 24, bold=True)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_TERRAIN = (50, 60, 70)
        self.COLOR_TRACK = (0, 180, 255)
        self.COLOR_TRACK_PEN_UP = (100, 120, 140)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_CHECKPOINT = (255, 220, 0)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_START = (50, 255, 50)
        self.COLOR_CURSOR = (255, 100, 200)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rider_pos = [0, 0]
        self.rider_vel = [0, 0]
        self.cursor_pos = [0, 0]
        self.is_pen_down = True
        self.track_points = []
        self.particles = []
        self.checkpoints = []
        self.finish_line = {}
        self.start_pos = (0, 0)
        self.terrain_points = []
        self.np_random = None
        self.last_reward_info = ""

        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_reward_info = ""
        
        self._generate_level()
        
        self.rider_pos = list(self.start_pos)
        self.rider_vel = [2.0, 0.0]
        
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.is_pen_down = True
        self.track_points = [[self.cursor_pos.copy()]]
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle player input
        self._handle_input(action)
        
        # 2. Update physics and game objects
        self._update_rider()
        self._update_particles()
        
        # 3. Calculate rewards
        reward += 0.1  # Survival reward

        checkpoint_reward = self._check_collectible_collisions()
        reward += checkpoint_reward
        if checkpoint_reward > 0:
            self.last_reward_info = f"+{int(checkpoint_reward)} CHECKPOINT!"

        # 4. Check for termination
        terminated, term_reward = self._check_termination()
        reward += term_reward
        if term_reward < 0:
            self.last_reward_info = f"{int(term_reward)} CRASHED!"
        elif self.win:
            self.last_reward_info = f"+{int(term_reward)} FINISH!"

        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        is_pen_down_now = not bool(space_held)
        
        if is_pen_down_now and not self.is_pen_down:
            # Pen just went down, start a new line strip
            self.track_points.append([self.cursor_pos.copy()])
        self.is_pen_down = is_pen_down_now

        # Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            new_x = self.cursor_pos[0] + dx * self.CURSOR_SPEED
            new_y = self.cursor_pos[1] + dy * self.CURSOR_SPEED
            self.cursor_pos[0] = np.clip(new_x, 0, self.WIDTH)
            self.cursor_pos[1] = np.clip(new_y, 0, self.HEIGHT)

            if self.is_pen_down:
                if not self.track_points: self.track_points.append([])
                last_point = self.track_points[-1][-1] if self.track_points[-1] else [-1e5, -1e5]
                dist_sq = (self.cursor_pos[0] - last_point[0])**2 + (self.cursor_pos[1] - last_point[1])**2
                if dist_sq > 4**2:
                     self.track_points[-1].append(self.cursor_pos.copy())

    def _update_rider(self):
        # Find closest track point
        closest_dist_sq = float('inf')
        closest_segment = None
        closest_point_on_segment = None
        
        for point_list in self.track_points:
            if len(point_list) < 2: continue
            for i in range(len(point_list) - 1):
                p1 = np.array(point_list[i])
                p2 = np.array(point_list[i+1])
                l2 = np.sum((p1 - p2)**2)
                if l2 == 0.0: continue
                
                t = max(0, min(1, np.dot(self.rider_pos - p1, p2 - p1) / l2))
                pt_on_seg = p1 + t * (p2 - p1)
                dist_sq = np.sum((self.rider_pos - pt_on_seg)**2)
                
                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
                    closest_segment = (p1, p2)
                    closest_point_on_segment = pt_on_seg

        on_track = closest_segment is not None and closest_dist_sq < (self.RIDER_RADIUS * 1.5)**2

        if on_track:
            # ON TRACK PHYSICS
            self.rider_pos = list(closest_point_on_segment)
            segment_vec = closest_segment[1] - closest_segment[0]
            seg_len = np.linalg.norm(segment_vec)
            if seg_len > 1e-6:
                segment_dir = segment_vec / seg_len
                current_speed = np.linalg.norm(self.rider_vel)
                dot_product = np.dot(self.rider_vel, segment_dir)
                
                # Use projected speed if moving along track, otherwise use current speed
                projected_speed = max(current_speed * 0.5, dot_product)
                
                slope_effect = segment_dir[1] * self.GRAVITY * 1.5
                new_speed = projected_speed + slope_effect
                new_speed *= 0.995 # Friction
                
                self.rider_vel = list(segment_dir * new_speed)
                
                if self.steps % 3 == 0 and new_speed > 1:
                    # Sound: # sled_slide.wav
                    self._create_particle(self.rider_pos, count=1, color=(100, 150, 200), life=15, size=3, speed=0.5)
        else:
            # AIRBORNE PHYSICS
            self.rider_vel[1] += self.GRAVITY
            self.rider_vel[0] *= 0.998
            self.rider_vel[1] *= 0.998
        
        self.rider_pos[0] += self.rider_vel[0]
        self.rider_pos[1] += self.rider_vel[1]
        
        # Terrain collision
        terrain_y = self._get_terrain_height(self.rider_pos[0])
        if self.rider_pos[1] > terrain_y - self.RIDER_RADIUS:
            self.rider_pos[1] = terrain_y - self.RIDER_RADIUS
            if self.rider_vel[1] > 2: # Hard impact
                # Sound: # terrain_impact.wav
                self._create_particle(self.rider_pos, count=5, color=self.COLOR_TERRAIN, life=20, size=4, speed=2)
            self.rider_vel[1] *= -0.3 # Bounce
            self.rider_vel[0] *= 0.8  # Friction

    def _get_terrain_height(self, x):
        if not self.terrain_points or x < self.terrain_points[0][0] or x > self.terrain_points[-1][0]:
            return self.HEIGHT + 100
        
        p1 = self.terrain_points[0]
        for i in range(len(self.terrain_points) - 1):
            if self.terrain_points[i][0] <= x < self.terrain_points[i+1][0]:
                p1 = self.terrain_points[i]
                p2 = self.terrain_points[i+1]
                t = (x - p1[0]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else 0
                return p1[1] + t * (p2[1] - p1[1])
        return self.terrain_points[-1][1]

    def _check_collectible_collisions(self):
        reward = 0
        for cp in self.checkpoints:
            if not cp['collected']:
                dist_sq = (self.rider_pos[0] - cp['pos'][0])**2 + (self.rider_pos[1] - cp['pos'][1])**2
                if dist_sq < (self.RIDER_RADIUS + 12)**2:
                    cp['collected'] = True
                    reward += 10
                    # Sound: # checkpoint_get.wav
                    self._create_particle(cp['pos'], count=30, color=self.COLOR_CHECKPOINT, life=40, size=6, speed=6)
        return reward

    def _check_termination(self):
        if not (0 < self.rider_pos[0] < self.WIDTH and -50 < self.rider_pos[1] < self.HEIGHT + 50):
            # Sound: # crash.wav
            return True, -100

        if self.steps >= self.MAX_STEPS:
            return True, -50

        all_cp_collected = all(cp['collected'] for cp in self.checkpoints)
        if all_cp_collected:
            dist_sq = (self.rider_pos[0] - self.finish_line['pos'][0])**2 + (self.rider_pos[1] - self.finish_line['pos'][1])**2
            if dist_sq < (self.RIDER_RADIUS + 12)**2:
                self.win = True
                # Sound: # win_level.wav
                self._create_particle(self.finish_line['pos'], count=80, color=self.COLOR_FINISH, life=80, size=8, speed=8)
                return True, 100
        
        return False, 0

    def _generate_level(self):
        self.terrain_points = []
        num_points = 32
        for i in range(num_points + 1):
            x = (self.WIDTH / num_points) * i
            y = self.HEIGHT * 0.75 + math.sin(i / 5) * 40 + self.np_random.uniform(-10, 10)
            self.terrain_points.append((x, y))

        self.start_pos = (50, self._get_terrain_height(50) - 80)
        
        self.checkpoints = [
            {'pos': (220, self.HEIGHT * 0.45), 'collected': False},
            {'pos': (450, self.HEIGHT * 0.5), 'collected': False}
        ]
        
        self.finish_line = {'pos': (self.WIDTH - 60, self._get_terrain_height(self.WIDTH - 60) - 80)}
        for cp in self.checkpoints: cp['collected'] = False

    def _create_particle(self, pos, count, color, life, size, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.5) * speed
            vel = [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(life//2, life),
                'max_life': life, 'color': color, 'size': self.np_random.uniform(size * 0.5, size)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Terrain
        pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, self.terrain_points + [(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)])

        # Track
        for point_list in self.track_points:
            if len(point_list) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, point_list, 2)
        
        # Start, Checkpoints, Finish
        self._draw_flag(self.start_pos, self.COLOR_START, 'S')
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if not cp['collected'] else (80, 70, 0)
            self._draw_flag(cp['pos'], color, str(self.checkpoints.index(cp) + 1))
        
        all_collected = all(c['collected'] for c in self.checkpoints)
        finish_color = self.COLOR_FINISH if all_collected else (80, 20, 20)
        self._draw_flag(self.finish_line['pos'], finish_color, 'F')

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((int(p['size']*2), int(p['size']*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Rider
        rider_x, rider_y = int(self.rider_pos[0]), int(self.rider_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # Cursor
        cursor_color = self.COLOR_CURSOR if self.is_pen_down else self.COLOR_TRACK_PEN_UP
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, cursor_x, cursor_y, 5, cursor_color)
        pygame.gfxdraw.aacircle(self.screen, cursor_x, cursor_y, 5, cursor_color)

    def _draw_flag(self, pos, color, text):
        x, y = int(pos[0]), int(pos[1])
        pygame.draw.line(self.screen, (200, 200, 200), (x, y), (x, y - 20), 2)
        flag_poly = [(x, y - 20), (x + 15, y - 15), (x, y - 10)]
        pygame.gfxdraw.filled_polygon(self.screen, flag_poly, color)
        pygame.gfxdraw.aapolygon(self.screen, flag_poly, color)
        if text:
            txt_surf = self.font_ui.render(text, True, (0,0,0))
            self.screen.blit(txt_surf, (x + 3, y - 21))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Checkpoints
        collected = sum(1 for cp in self.checkpoints if cp['collected'])
        total = len(self.checkpoints)
        cp_text = self.font_ui.render(f"CHECKPOINTS: {collected}/{total}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cp_text, (10, 35))

        # Reward message
        if self.last_reward_info and self.steps < self.MAX_STEPS:
            msg_surf = self.font_msg.render(self.last_reward_info, True, self.COLOR_CHECKPOINT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, 40))
            self.screen.blit(msg_surf, msg_rect)
        
        # Game Over / Win message
        if self.game_over:
            msg = "FINISH!" if self.win else "GAME OVER"
            color = self.COLOR_FINISH if self.win else self.COLOR_UI_TEXT
            end_surf = self.font_msg.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "checkpoints_collected": sum(1 for cp in self.checkpoints if cp['collected']),
            "win": self.win,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    import os
    # For headless execution, uncomment the following line:
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play with keyboard ---
    pygame.display.set_caption("Sled Rider")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30)

    env.close()