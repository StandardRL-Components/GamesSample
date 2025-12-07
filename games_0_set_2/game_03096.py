import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to draw a track line. Shift to reset cursor to last point."
    )

    game_description = (
        "Draw a track for a sled to navigate a physics-based course. Reach checkpoints and the finish line as fast as possible."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_SLED = (255, 220, 0)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_TERRAIN = (15, 25, 45)
        self.COLOR_START = (0, 255, 0, 50)
        self.COLOR_FINISH = (255, 0, 0, 50)
        self.COLOR_CHECKPOINT = (0, 180, 255, 50)
        self.COLOR_CURSOR = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 240)

        # Game constants
        self.MAX_STEPS = 500
        self.MAX_TIME = 120.0
        self.CURSOR_SPEED = 10
        self.SLED_RADIUS = 6
        self.GRAVITY = pygame.math.Vector2(0, 0.08)
        self.FRICTION = 0.99
        self.DAMPING = 0.998
        self.COLLISION_ELASTICITY = 0.7

        # Pre-render background
        self.background = self._create_gradient_background()

        # Define stage data
        self.stage_data = self._define_stages()

        # Initialize state variables
        self.reset()
        
        # Run self-check
        # self.validate_implementation() # Commented out for submission

    def _define_stages(self):
        stages = [
            { # Stage 1: Simple slope
                "start_pos": pygame.math.Vector2(50, 50),
                "finish_area": pygame.Rect(self.WIDTH - 100, self.HEIGHT - 80, 80, 80),
                "checkpoints": [],
                "terrain": [
                    [(0, 100), (100, 150), (200, 120), (0, 200)],
                    [(self.WIDTH - 200, self.HEIGHT - 50), (self.WIDTH, self.HEIGHT - 100), (self.WIDTH, self.HEIGHT), (self.WIDTH - 250, self.HEIGHT)],
                ]
            },
            { # Stage 2: A gap to cross
                "start_pos": pygame.math.Vector2(50, 50),
                "finish_area": pygame.Rect(self.WIDTH - 100, 50, 80, 80),
                "checkpoints": [pygame.Rect(180, self.HEIGHT - 80, 60, 60)],
                "terrain": [
                    [(0, 100), (200, 80), (200, self.HEIGHT), (0, self.HEIGHT)],
                    [(self.WIDTH - 200, 100), (self.WIDTH, 80), (self.WIDTH, self.HEIGHT), (self.WIDTH - 200, self.HEIGHT)],
                ]
            },
            { # Stage 3: A loop challenge
                "start_pos": pygame.math.Vector2(50, self.HEIGHT - 50),
                "finish_area": pygame.Rect(self.WIDTH - 100, self.HEIGHT - 50, 80, 50),
                "checkpoints": [pygame.Rect(self.WIDTH / 2 - 30, 50, 60, 60)],
                "terrain": [
                    [(200, self.HEIGHT), (250, 150), (self.WIDTH-250, 150), (self.WIDTH-200, self.HEIGHT)],
                ]
            }
        ]
        return stages

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color = [
                self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * y / self.HEIGHT
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _load_stage(self, stage_index):
        if stage_index >= len(self.stage_data):
            self.game_won = True
            return

        stage = self.stage_data[stage_index]
        self.current_stage = stage_index
        self.start_pos = stage["start_pos"].copy()
        self.finish_area = stage["finish_area"]
        self.checkpoints = stage["checkpoints"]
        self.terrain = stage["terrain"]
        
        self.sled_pos = self.start_pos.copy()
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.track_lines = []
        self.cursor_start = self.start_pos.copy()
        self.cursor_end = self.start_pos.copy()
        self.checkpoints_cleared = [False] * len(self.checkpoints)
        self.on_ground = False
        self.last_checkpoint_y = self.start_pos.y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_elapsed = 0.0
        self.game_over = False
        self.game_won = False
        self.win_message = ""
        self.particles = []
        
        self._load_stage(0)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update drawing cursor
        if movement == 1: self.cursor_end.y -= self.CURSOR_SPEED # Up
        elif movement == 2: self.cursor_end.y += self.CURSOR_SPEED # Down
        elif movement == 3: self.cursor_end.x -= self.CURSOR_SPEED # Left
        elif movement == 4: self.cursor_end.x += self.CURSOR_SPEED # Right
        
        self.cursor_end.x = np.clip(self.cursor_end.x, 0, self.WIDTH)
        self.cursor_end.y = np.clip(self.cursor_end.y, 0, self.HEIGHT)

        if shift_held:
            self.cursor_end = self.cursor_start.copy()

        reward = 0
        terminated = False
        
        # 2. Place line and run physics simulation
        if space_held and self.cursor_start.distance_to(self.cursor_end) > 1:
            # Add a new line segment to the track
            new_line = (self.cursor_start.copy(), self.cursor_end.copy())
            self.track_lines.append(new_line)
            self.cursor_start = self.cursor_end.copy()
            
            # Run simulation
            sim_reward, sim_terminated, sim_info = self._run_physics_simulation()
            reward += sim_reward
            terminated = sim_terminated
            
            if "crashed" in sim_info:
                # sfx: explosion
                self._spawn_crash_particles(self.sled_pos)
                reward -= 50
                self.win_message = "CRASHED!"
            elif "finished_stage" in sim_info:
                # sfx: success chime
                reward += 10 * (len(self.stage_data) - self.current_stage) # Bonus for later stages
                self._load_stage(self.current_stage + 1)
                if self.game_won:
                    # sfx: victory fanfare
                    reward += 100
                    reward -= 0.01 * self.time_elapsed # Speed bonus
                    self.win_message = f"ALL STAGES CLEAR! Time: {self.time_elapsed:.2f}s"
                    terminated = True
            elif "finished_checkpoint" in sim_info:
                # sfx: checkpoint sound
                reward += 10
        else:
            # Small penalty for thinking time (taking a step without drawing)
            reward -= 0.05

        self.steps += 1
        self.score += reward
        
        # 3. Check for other termination conditions
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.win_message = "MAX STEPS REACHED"
        if self.time_elapsed >= self.MAX_TIME:
            terminated = True
            self.win_message = "TIME LIMIT REACHED"
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _run_physics_simulation(self):
        sim_steps = 150
        total_reward = 0
        terminated = False
        info = {}

        for _ in range(sim_steps):
            if terminated: break

            self.time_elapsed += 1 / 60.0 # Assuming 60 physics ticks per second
            
            # Update sled physics
            prev_pos = self.sled_pos.copy()
            self.sled_vel += self.GRAVITY
            self.sled_vel *= self.DAMPING
            self.sled_pos += self.sled_vel
            
            self.on_ground = False
            
            # Collision with track
            for line_start, line_end in self.track_lines:
                self._handle_line_collision(line_start, line_end)
            
            # Collision with terrain
            if self._handle_terrain_collision():
                terminated = True
                info["crashed"] = True
                break
                
            if not self.on_ground:
                # Penalty for being airborne and below last checkpoint
                if self.sled_pos.y > self.last_checkpoint_y + self.SLED_RADIUS * 5:
                    total_reward -= 0.02
            else:
                total_reward += 0.01

            # Check for out of bounds
            if not (0 < self.sled_pos.x < self.WIDTH and 0 < self.sled_pos.y < self.HEIGHT):
                terminated = True
                info["crashed"] = True
                break

            # Check for checkpoints
            for i, chk in enumerate(self.checkpoints):
                if not self.checkpoints_cleared[i] and chk.collidepoint(self.sled_pos):
                    self.checkpoints_cleared[i] = True
                    info["finished_checkpoint"] = True
                    self.last_checkpoint_y = chk.centery
                    break # only one checkpoint per sim run

            # Check for finish line
            if all(self.checkpoints_cleared) and self.finish_area.collidepoint(self.sled_pos):
                info["finished_stage"] = True
                break
                
        return total_reward, terminated, info

    def _handle_line_collision(self, p1, p2):
        p1_vec = pygame.math.Vector2(p1)
        p2_vec = pygame.math.Vector2(p2)
        line_vec = p2_vec - p1_vec
        line_len_sq = line_vec.length_squared()

        if line_len_sq == 0: return

        t = ((self.sled_pos - p1_vec).dot(line_vec)) / line_len_sq
        t = np.clip(t, 0, 1)
        
        closest_point = p1_vec + t * line_vec
        dist_vec = self.sled_pos - closest_point
        dist = dist_vec.length()

        if dist < self.SLED_RADIUS:
            self.on_ground = True
            # Collision response
            penetration = self.SLED_RADIUS - dist
            
            # Check for zero distance to avoid normalization error
            if dist > 0:
                normal = dist_vec.normalize()
                self.sled_pos += normal * penetration
                
                reflected_vel = self.sled_vel.reflect(normal) * self.COLLISION_ELASTICITY
                
                # Apply friction
                tangent = pygame.math.Vector2(-normal.y, normal.x)
                friction_force = tangent * (self.sled_vel.dot(tangent)) * (1 - self.FRICTION)
                
                self.sled_vel = reflected_vel - friction_force

    def _handle_terrain_collision(self):
        # Simple AABB check for performance before precise check
        sled_rect = pygame.Rect(self.sled_pos.x - self.SLED_RADIUS, self.sled_pos.y - self.SLED_RADIUS, self.SLED_RADIUS*2, self.SLED_RADIUS*2)
        for poly_pts in self.terrain:
            poly_rect = pygame.Rect(poly_pts[0], (0,0)).unionall([pygame.Rect(p, (0,0)) for p in poly_pts])
            if not sled_rect.colliderect(poly_rect):
                continue
            
            # More precise check: point inside polygon or circle intersects edge
            if self._is_point_in_polygon(self.sled_pos, poly_pts):
                return True
            for i in range(len(poly_pts)):
                p1 = poly_pts[i]
                p2 = poly_pts[(i + 1) % len(poly_pts)]
                # Create a fake line to reuse collision logic, but just for detection
                if self._line_circle_intersect(p1, p2, self.sled_pos, self.SLED_RADIUS):
                    return True
        return False
    
    def _is_point_in_polygon(self, point, polygon):
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _line_circle_intersect(self, p1, p2, circle_center, radius):
        p1_vec = pygame.math.Vector2(p1)
        p2_vec = pygame.math.Vector2(p2)
        line_vec = p2_vec - p1_vec
        line_len_sq = line_vec.length_squared()
        if line_len_sq == 0: return circle_center.distance_to(p1_vec) < radius
        t = ((circle_center - p1_vec).dot(line_vec)) / line_len_sq
        t = np.clip(t, 0, 1)
        closest_point = p1_vec + t * line_vec
        return circle_center.distance_to(closest_point) < radius

    def _spawn_crash_particles(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.uniform(0.5, 1.5),
                'size': random.uniform(2, 5),
                'color': random.choice([self.COLOR_SLED, (255,150,0), (200,200,200)])
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time": self.time_elapsed,
            "stage": self.current_stage + 1
        }

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stage geometry
        for chk in self.checkpoints:
            s = pygame.Surface(chk.size, pygame.SRCALPHA)
            s.fill(self.COLOR_CHECKPOINT)
            self.screen.blit(s, chk.topleft)
        
        s = pygame.Surface(self.finish_area.size, pygame.SRCALPHA)
        s.fill(self.COLOR_FINISH)
        self.screen.blit(s, self.finish_area.topleft)

        for poly in self.terrain:
            pygame.gfxdraw.filled_polygon(self.screen, poly, self.COLOR_TERRAIN)
            pygame.gfxdraw.aapolygon(self.screen, poly, self.COLOR_TERRAIN)

        # Track
        for p1, p2 in self.track_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)
            
        # Preview line and cursor
        if not self.game_over:
            pygame.draw.aaline(self.screen, self.COLOR_CURSOR, self.cursor_start, self.cursor_end)
            pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_end.x), int(self.cursor_end.y), 4, self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_end.x), int(self.cursor_end.y), 4, self.COLOR_CURSOR)

        # Sled
        sled_x, sled_y = int(self.sled_pos.x), int(self.sled_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, sled_x, sled_y, self.SLED_RADIUS, self.COLOR_SLED)
        
        # Particles
        for p in self.particles[:]:
            p['vel'] += self.GRAVITY * 0.5
            p['pos'] += p['vel']
            p['life'] -= 1/60.0
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(max(0, min(255, p['life'] * 255)))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'] + (alpha,))

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        time_text = self.font_small.render(f"Time: {self.time_elapsed:.2f}s / {self.MAX_TIME:.0f}s", True, self.COLOR_UI_TEXT)
        stage_text = self.font_small.render(f"Stage: {self.current_stage + 1}/{len(self.stage_data)}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        self.screen.blit(stage_text, (self.WIDTH / 2 - stage_text.get_width() / 2, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Sled Drawer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The game logic is turn-based, so we only need to control the key polling rate
        clock.tick(30)

    env.close()