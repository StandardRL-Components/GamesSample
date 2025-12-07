
# Generated: 2025-08-27T19:31:08.349440
# Source Brief: brief_02180.md
# Brief Index: 2180

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Snail:
    def __init__(self, color, x, y, is_player=False, np_random=None):
        self.np_random = np_random if np_random is not None else np.random.default_rng()
        self.is_player = is_player
        self.base_color = color
        self.shell_color = tuple(max(0, c - 40) for c in color)
        self.outline_color = (255, 255, 100) if is_player else (0, 0, 0)
        
        self.initial_pos = [x, y]
        self.reset(x, y)

    def reset(self, x, y):
        self.pos = [float(x), float(y)]
        self.vel = [0.0, 0.0]
        self.size = 12 if self.is_player else 11
        self.bob_angle = 0
        self.is_boosting = False
        self.is_braking = False
        self.is_touching_edge = False
        self.ai_target_y_offset = 0
        self.ai_steer_timer = 0
        self.max_speed = 6.0
        self.turn_speed = 2.0

    def apply_action(self, movement, space_held, shift_held):
        self.is_boosting = False
        self.is_braking = False

        # Precision mode (Shift)
        if shift_held:
            accel = 0.5
            turn_authority = 1.5
            self.vel[0] *= 0.9  # Strong braking
            self.is_braking = True
        # Boost mode (Space)
        elif space_held and self.vel[0] > 1.0:
            accel = 6.0
            turn_authority = 0.3 # Harder to turn while boosting
            self.is_boosting = True
            # sfx: boost sound
        # Normal mode
        else:
            accel = 1.0
            turn_authority = 1.0
        
        # Movement
        if movement == 1:  # Up (Accelerate)
            self.vel[0] += accel * 0.2
        elif movement == 2:  # Down (Brake)
            self.vel[0] -= accel * 0.3
            self.is_braking = True
        
        if movement == 3:  # Left (Steer Up)
            self.vel[1] -= self.turn_speed * turn_authority * 0.2
        elif movement == 4:  # Right (Steer Down)
            self.vel[1] += self.turn_speed * turn_authority * 0.2

    def ai_update(self, track_center_y):
        # Basic AI: try to stay in the center of the track
        target_y = track_center_y + self.ai_target_y_offset
        
        # Accelerate
        self.vel[0] += 0.15

        # Steer towards target
        if self.pos[1] < target_y:
            self.vel[1] += self.turn_speed * 0.15
        else:
            self.vel[1] -= self.turn_speed * 0.15
        
        # Periodically change target y offset for variation
        self.ai_steer_timer -= 1
        if self.ai_steer_timer <= 0:
            self.ai_target_y_offset = self.np_random.uniform(-20, 20)
            self.ai_steer_timer = self.np_random.integers(30, 90)

    def update(self, track_center_y, track_half_width):
        # Apply physics
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Apply friction/drag
        self.vel[0] *= 0.98  # Horizontal drag
        self.vel[1] *= 0.90  # Vertical drag (harder to change lanes)
        self.vel[0] = max(0, min(self.vel[0], self.max_speed * (1.5 if self.is_boosting else 1.0)))

        # Animation
        self.bob_angle += self.vel[0] * 0.1

        # Track collision
        self.is_touching_edge = False
        track_top = track_center_y - track_half_width
        track_bottom = track_center_y + track_half_width
        
        if self.pos[1] - self.size < track_top:
            self.pos[1] = track_top + self.size
            self.vel[1] *= -0.5 # Bounce
            self.vel[0] *= 0.95 # Speed penalty
            self.is_touching_edge = True
            # sfx: bump
        if self.pos[1] + self.size > track_bottom:
            self.pos[1] = track_bottom - self.size
            self.vel[1] *= -0.5 # Bounce
            self.vel[0] *= 0.95 # Speed penalty
            self.is_touching_edge = True
            # sfx: bump

        # Check for falling off
        fall_margin = self.size * 2
        if self.pos[1] < track_top - fall_margin or self.pos[1] > track_bottom + fall_margin:
            return False # Fell off
        return True # Still on track

    def draw(self, surface, camera_x, particles):
        x, y = int(self.pos[0] - camera_x), int(self.pos[1])
        
        # Add particles for trail
        if self.vel[0] > 3.0 or self.is_boosting:
            p_color = (255, 255, 255, 150) if not self.is_boosting else (255, 200, 0, 200)
            p_vel = [-self.vel[0] * 0.5, self.np_random.uniform(-0.5, 0.5)]
            particles.append({'pos': [self.pos[0], self.pos[1]], 'vel': p_vel, 'life': 15, 'size': self.size / 2, 'color': p_color})

        # Bobbing animation
        bob = math.sin(self.bob_angle) * 2
        
        # Shell
        shell_pos = (x, int(y - self.size * 0.2 + bob))
        pygame.gfxdraw.aacircle(surface, shell_pos[0], shell_pos[1], int(self.size), self.shell_color)
        pygame.gfxdraw.filled_circle(surface, shell_pos[0], shell_pos[1], int(self.size), self.shell_color)
        
        # Shell spiral
        for i in range(5):
            angle_start = self.bob_angle * 0.5 + i * math.pi * 0.4
            angle_end = self.bob_angle * 0.5 + (i + 0.5) * math.pi * 0.4
            radius = self.size * (1 - i * 0.2)
            p1 = (shell_pos[0] + int(math.cos(angle_start) * radius), shell_pos[1] + int(math.sin(angle_start) * radius))
            p2 = (shell_pos[0] + int(math.cos(angle_end) * radius), shell_pos[1] + int(math.sin(angle_end) * radius))
            pygame.draw.line(surface, self.base_color, p1, p2, 2)

        # Body
        body_rect = pygame.Rect(x - self.size * 1.2, y + self.size * 0.1 + bob, self.size * 1.5, self.size * 0.8)
        pygame.draw.rect(surface, self.base_color, body_rect, border_radius=5)
        
        # Eye
        eye_pos = (int(x - self.size * 0.8), int(y + self.size * 0.3 + bob))
        pygame.draw.circle(surface, (255, 255, 255), eye_pos, 3)
        pygame.draw.circle(surface, (0, 0, 0), eye_pos, 1)

        # Player outline
        if self.is_player:
            pygame.gfxdraw.aacircle(surface, shell_pos[0], shell_pos[1], int(self.size) + 2, self.outline_color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Press space for a speed burst and hold shift for tight turns."
    game_description = "A vibrant, side-scrolling snail race. Outpace your rivals across three challenging stages to become the champion!"
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TRACK_LENGTH = 6000
        self.TRACK_WIDTH = 120
        self.TIME_LIMIT = 60.0

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        self.COLOR_BG = (135, 206, 235)
        self.COLOR_TRACK = (107, 142, 35)
        self.COLOR_TRACK_LINE = (255, 255, 255)
        self.COLOR_HILL_1 = (34, 139, 34)
        self.COLOR_HILL_2 = (0, 100, 0)
        
        self.reset()
        self.validate_implementation()
    
    def _generate_track(self):
        self.track_points = []
        points = 200
        for i in range(points + 1):
            x = (i / points) * self.TRACK_LENGTH
            y_base = self.HEIGHT / 2
            y_wave1 = math.sin(i / 20) * self.HEIGHT / 4
            y_wave2 = math.sin(i / 35) * self.HEIGHT / 6
            y = y_base + y_wave1 + y_wave2
            self.track_points.append((x, y))

    def _get_track_y(self, x):
        if x <= 0: return self.track_points[0][1]
        if x >= self.TRACK_LENGTH: return self.track_points[-1][1]
        
        segment_len = self.TRACK_LENGTH / (len(self.track_points) - 1)
        idx = int(x / segment_len)
        p1 = self.track_points[idx]
        p2 = self.track_points[idx + 1]
        
        interp = (x - p1[0]) / (p2[0] - p1[0])
        return p1[1] + interp * (p2[1] - p1[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.stage = 1
        self.score = 0
        self.time_left = self.TIME_LIMIT
        self.game_over = False
        self.game_won = False
        self.message = ""
        self.message_timer = 0
        self.camera_x = 0
        self.particles = []

        self._generate_track()
        self._reset_stage()
        
        return self._get_observation(), self._get_info()

    def _reset_stage(self):
        self.time_left = self.TIME_LIMIT
        self.camera_x = 0
        start_y = self._get_track_y(150)

        if not hasattr(self, 'player_snail'):
            self.player_snail = Snail((255, 80, 80), 150, start_y, True, self.np_random)
            self.ai_snails = [
                Snail((80, 80, 255), 140, start_y - 30, np_random=self.np_random),
                Snail((255, 255, 80), 130, start_y + 30, np_random=self.np_random)
            ]
        else:
            self.player_snail.reset(150, start_y)
            self.ai_snails[0].reset(140, start_y - 30)
            self.ai_snails[1].reset(130, start_y + 30)

        self._update_ai_speed()
        self.last_ranks = self._get_ranks()
        self.message = f"STAGE {self.stage}"
        self.message_timer = self.FPS * 2

    def _update_ai_speed(self):
        base_speed = 5.5 + self.stage * 0.2
        for ai in self.ai_snails:
            ai.max_speed = base_speed + self.np_random.uniform(-0.2, 0.2)

    def _get_ranks(self):
        all_snails = [self.player_snail] + self.ai_snails
        indexed_pos = [(i, snail.pos[0]) for i, snail in enumerate(all_snails)]
        indexed_pos.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in indexed_pos]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Update game state
        self.time_left -= 1.0 / self.FPS
        if self.message_timer > 0:
            self.message_timer -= 1

        # 2. Apply actions and update entities
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.player_snail.apply_action(movement, space_held, shift_held)

        all_snails = [self.player_snail] + self.ai_snails
        for i, snail in enumerate(all_snails):
            track_center_y = self._get_track_y(snail.pos[0])
            if not snail.is_player:
                snail.ai_update(track_center_y)
            
            on_track = snail.update(track_center_y, self.TRACK_WIDTH / 2)
            if not on_track and snail.is_player:
                self.game_over = True
                self.message = "FELL OFF TRACK!"
                self.message_timer = self.FPS * 5
                reward -= 100
                # sfx: falling
        
        self._update_particles()

        # 3. Calculate rewards
        # Forward movement reward
        reward += self.player_snail.vel[0] * 0.01
        # Boundary hit penalty
        if self.player_snail.is_touching_edge:
            reward -= 0.2
        # Overtaking reward/penalty
        current_ranks = self._get_ranks()
        player_old_rank = self.last_ranks.index(0)
        player_new_rank = current_ranks.index(0)
        if player_new_rank < player_old_rank:
            reward += 10 # Overtook
        elif player_new_rank > player_old_rank:
            reward -= 10 # Was overtaken
        self.last_ranks = current_ranks

        self.score += reward
        
        # 4. Check for termination/progression
        terminated = self.game_over

        # Finish line logic
        if self.player_snail.pos[0] >= self.TRACK_LENGTH and not self.game_over:
            player_rank_final = self._get_ranks().index(0)
            if player_rank_final == 0: # Player won
                if self.stage == 3:
                    reward += 300
                    self.game_over = True
                    self.game_won = True
                    self.message = "RACE CHAMPION!"
                    self.message_timer = self.FPS * 5
                    # sfx: final win
                else:
                    reward += 100
                    self.stage += 1
                    self._reset_stage()
                    # sfx: stage win
            else: # Player lost
                self.game_over = True
                self.message = f"FINISHED {player_rank_final + 1}nd!"
                self.message_timer = self.FPS * 5
                reward -= 50
            terminated = self.game_over

        # Time out logic
        if self.time_left <= 0 and not self.game_over:
            self.game_over = True
            self.message = "TIME'S UP!"
            self.message_timer = self.FPS * 5
            reward -= 100
            terminated = True
            # sfx: lose
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update camera
        all_snails = [self.player_snail] + self.ai_snails
        leading_x = max(s.pos[0] for s in all_snails)
        target_cam_x = leading_x - self.WIDTH / 3
        self.camera_x += (target_cam_x - self.camera_x) * 0.1 # Smooth camera

        # Draw background hills (parallax)
        for i in range(3):
            offset = (self.camera_x * (0.1 * (i + 1))) % self.WIDTH
            pygame.draw.rect(self.screen, [self.COLOR_HILL_2, self.COLOR_HILL_1, self.COLOR_HILL_1][i], (0 - offset, self.HEIGHT - 150 + i * 40, self.WIDTH, 150))
            pygame.draw.rect(self.screen, [self.COLOR_HILL_2, self.COLOR_HILL_1, self.COLOR_HILL_1][i], (self.WIDTH - offset, self.HEIGHT - 150 + i * 40, self.WIDTH, 150))

        # Draw track
        track_poly = []
        finish_line_x = None
        for x_screen in range(0, self.WIDTH + 20, 20):
            x_world = self.camera_x + x_screen
            if x_world >= self.TRACK_LENGTH and finish_line_x is None:
                finish_line_x = x_screen
            y_world = self._get_track_y(x_world)
            track_poly.append((x_screen, y_world - self.TRACK_WIDTH / 2))
        for x_screen in range(self.WIDTH + 20, 0, -20):
            x_world = self.camera_x + x_screen
            y_world = self._get_track_y(x_world)
            track_poly.append((x_screen, y_world + self.TRACK_WIDTH / 2))
        
        if len(track_poly) > 2:
            pygame.gfxdraw.aapolygon(self.screen, track_poly, self.COLOR_TRACK)
            pygame.gfxdraw.filled_polygon(self.screen, track_poly, self.COLOR_TRACK)

        # Draw track center line
        for x_screen in range(0, self.WIDTH, 20):
            x_world = self.camera_x + x_screen
            if int(x_world / 40) % 2 == 0:
                y_start = self._get_track_y(x_world)
                y_end = self._get_track_y(x_world + 20)
                x_end = x_screen + 20
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (x_screen, y_start), (x_end, y_end), 3)

        # Draw finish line
        if finish_line_x is not None:
            y_world = self._get_track_y(self.TRACK_LENGTH)
            for i in range(10):
                color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
                h = self.TRACK_WIDTH / 10
                rect = pygame.Rect(finish_line_x, y_world - self.TRACK_WIDTH / 2 + i * h, 10, h)
                pygame.draw.rect(self.screen, color, rect)

        # Draw particles
        for p in self.particles:
            x, y = int(p['pos'][0] - self.camera_x), int(p['pos'][1])
            alpha_color = p['color'][:3] + (int(p['color'][3] * (p['life'] / 15)),)
            if p['size'] > 1:
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, alpha_color, (p['size'], p['size']), p['size'])
                self.screen.blit(s, (x - p['size'], y - p['size']))

        # Draw snails
        all_snails_sorted = sorted(all_snails, key=lambda s: s.pos[1])
        for snail in all_snails_sorted:
            snail.draw(self.screen, self.camera_x, self.particles)

    def _render_ui(self):
        # Stage indicator
        stage_text = self.font_ui.render(f"Stage: {self.stage}/3", True, (0, 0, 0))
        self.screen.blit(stage_text, (10, 10))
        
        # Timer
        time_color = (200, 0, 0) if self.time_left < 10 else (0, 0, 0)
        time_text = self.font_ui.render(f"Time: {max(0, self.time_left):.1f}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, (0, 0, 0))
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Big message text
        if self.message and self.message_timer > 0:
            color = (0, 200, 0) if self.game_won else ((200, 0, 0) if self.game_over else (255, 255, 255))
            msg_surf = self.font_msg.render(self.message, True, color)
            outline_surf = self.font_msg.render(self.message, True, (0,0,0))
            x = self.WIDTH // 2 - msg_surf.get_width() // 2
            y = self.HEIGHT // 2 - msg_surf.get_height() // 2
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                self.screen.blit(outline_surf, (x + dx, y + dy))
            self.screen.blit(msg_surf, (x, y))

    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "time_left": round(self.time_left, 2),
            "player_x": self.player_snail.pos[0],
            "player_rank": self.last_ranks.index(0) + 1 if 0 in self.last_ranks else 3
        }
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame Interactive Player ---
    
    # Remap keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: (1, 0, 0),
        pygame.K_DOWN: (2, 0, 0),
        pygame.K_LEFT: (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snail Race")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    while not done:
        # Action defaults to no-op
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Handle simultaneous presses
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        
        if keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
            
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            done = False
            
        clock.tick(env.FPS)
        
    pygame.quit()