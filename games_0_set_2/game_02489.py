
# Generated: 2025-08-28T05:02:43.958964
# Source Brief: brief_02489.md
# Brief Index: 2489

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim, SHIFT to decrease power, SPACE to increase power. Press ↑ to shoot."
    )

    game_description = (
        "A vibrant isometric mini-golf game. Aim your shot, manage your power, and navigate tricky courses. "
        "Complete all 9 holes in under 10 strokes to win."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_GRASS = (58, 156, 100)
    COLOR_GRASS_DARK = (50, 135, 86)
    COLOR_WALL = (100, 110, 120)
    COLOR_WALL_TOP = (120, 130, 140)
    COLOR_BALL = (255, 255, 255)
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_HOLE = (20, 20, 20)
    COLOR_FLAG_POLE = (200, 200, 200)
    COLOR_FLAG = (220, 50, 50)
    COLOR_AIM = (50, 150, 255, 150)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 100)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STROKES = 10
    NUM_HOLES = 9
    MAX_EPISODE_STEPS = 1000
    
    # Physics
    FRICTION = 0.97
    POWER_MULTIPLIER = 0.25
    MIN_POWER = 5
    MAX_POWER = 100
    AIM_SENSITIVITY = 0.05
    POWER_SENSITIVITY = 2
    STOP_THRESHOLD = 0.1
    STUCK_VELOCITY_THRESHOLD = 0.2
    STUCK_FRAMES_LIMIT = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self._courses = self._generate_courses()
        self.particles = []

        self.game_state = 'AIMING'
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.aim_angle = 0
        self.shot_power = self.MIN_POWER
        self.current_hole_index = 0
        self.stroke_count = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_dist_to_hole = 0
        self.stuck_counter = 0

        self.reset()
        
        self.validate_implementation()

    def _generate_courses(self):
        courses = []
        # Hole 1: Straight shot
        courses.append({
            "start": (100, 200), "hole": (500, 200), "radius": 10,
            "walls": [((50, 150), (550, 150)), ((50, 250), (550, 250))]
        })
        # Hole 2: Dogleg right
        courses.append({
            "start": (100, 100), "hole": (500, 300), "radius": 10,
            "walls": [((50, 50), (350, 50)), ((350, 50), (350, 250)), ((50, 150), (300, 150)), 
                      ((300, 200), (550, 200)), ((300, 350), (550, 350))]
        })
        # Hole 3: Ricochet
        courses.append({
            "start": (100, 200), "hole": (500, 200), "radius": 10,
            "walls": [((50, 150), (550, 150)), ((50, 250), (550, 250)), ((300, 150), (300, 180)), ((300, 220), (300, 250))]
        })
        # Hole 4: Funnel
        courses.append({
            "start": (100, 200), "hole": (540, 200), "radius": 8,
            "walls": [((50, 100), (500, 170)), ((50, 300), (500, 230))]
        })
        # Hole 5: Obstacle course
        courses.append({
            "start": (60, 200), "hole": (580, 200), "radius": 10,
            "walls": [((40, 150), (600, 150)), ((40, 250), (600, 250)),
                      ((150, 150), (150, 190)), ((150, 210), (150, 250)),
                      ((300, 150), (300, 250)),
                      ((450, 150), (450, 190)), ((450, 210), (450, 250))]
        })
        # Hole 6: The 'S'
        courses.append({
            "start": (100, 100), "hole": (100, 300), "radius": 10,
            "walls": [((50, 50), (400, 50)), ((50, 150), (400, 150)),
                      ((150, 200), (500, 200)), ((150, 350), (500, 350))]
        })
        # Hole 7: Long shot with side walls
        courses.append({
            "start": (100, 200), "hole": (500, 200), "radius": 10,
            "walls": [((80, 180), (520, 180)), ((80, 220), (520, 220))]
        })
        # Hole 8: Maze-like
        courses.append({
            "start": (60, 60), "hole": (580, 340), "radius": 10,
            "walls": [((40, 40), (600, 40)), ((40, 40), (40, 360)), ((600, 40), (600, 360)), ((40, 360), (600, 360)),
                      ((40, 150), (400, 150)), ((200, 250), (600, 250))]
        })
        # Hole 9: The Final Challenge
        courses.append({
            "start": (320, 350), "hole": (320, 50), "radius": 8,
            "walls": [((200, 20), (440, 20)), ((200, 380), (440, 380)),
                      ((200, 20), (200, 150)), ((440, 20), (440, 150)),
                      ((200, 250), (200, 380)), ((440, 250), (440, 380)),
                      ((250, 200), (390, 200))]
        })
        return courses

    def _load_hole(self, hole_index):
        if hole_index >= len(self._courses):
            self.game_over = True
            return

        course = self._courses[hole_index]
        self.ball_pos = pygame.Vector2(course["start"])
        self.ball_vel = pygame.Vector2(0, 0)
        self.current_course = course
        self.game_state = 'AIMING'
        self.aim_angle = math.atan2(course["hole"][1] - course["start"][1], course["hole"][0] - course["start"][0])
        self.shot_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.last_dist_to_hole = self.ball_pos.distance_to(pygame.Vector2(course["hole"]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stroke_count = 0
        self.current_hole_index = 0
        self.game_over = False
        self.particles.clear()
        
        self._load_hole(self.current_hole_index)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.game_state == 'AIMING':
            # Adjust Aim
            if movement == 3:  # Left
                self.aim_angle -= self.AIM_SENSITIVITY
            elif movement == 4:  # Right
                self.aim_angle += self.AIM_SENSITIVITY
            
            # Adjust Power
            if space_held:
                self.shot_power += self.POWER_SENSITIVITY
            if shift_held:
                self.shot_power -= self.POWER_SENSITIVITY
            self.shot_power = max(self.MIN_POWER, min(self.MAX_POWER, self.shot_power))

            # Shoot
            if movement == 1:  # Up
                self.game_state = 'BALL_MOVING'
                self.stroke_count += 1
                power = self.shot_power * self.POWER_MULTIPLIER
                self.ball_vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * power
                
                # --- Physics Simulation within one step ---
                reward -= 0.1 # Cost per stroke
                self.stuck_counter = 0
                
                physics_steps = 0
                while self.ball_vel.magnitude() > self.STOP_THRESHOLD and physics_steps < 500:
                    
                    self.ball_pos += self.ball_vel
                    self.ball_vel *= self.FRICTION
                    
                    # Stuck check
                    if self.ball_vel.magnitude() < self.STUCK_VELOCITY_THRESHOLD:
                        self.stuck_counter += 1
                    else:
                        self.stuck_counter = 0

                    if self.stuck_counter > self.STUCK_FRAMES_LIMIT:
                        self.stroke_count += 1 # Penalty stroke
                        self._create_particles(self.ball_pos, (255, 255, 0), 20, 2) # Penalty effect
                        self._load_hole(self.current_hole_index)
                        reward -= 5 # Penalty reward
                        break # Exit physics loop

                    # Wall collisions
                    for wall in self.current_course["walls"]:
                        hit, self.ball_pos, self.ball_vel = self._collide_line_circle(wall[0], wall[1], self.ball_pos, 8, self.ball_vel)
                        if hit:
                            reward -= 1 # Obstacle hit penalty
                            self._create_particles(self.ball_pos, self.COLOR_WALL_TOP, 10)
                            # sfx: wall_thud.wav

                    # Hole check
                    hole_pos = pygame.Vector2(self.current_course["hole"])
                    if self.ball_pos.distance_to(hole_pos) < self.current_course["radius"] and self.ball_vel.magnitude() < 2.5:
                        reward += 10 # Sunk ball reward
                        self._create_particles(hole_pos, self.COLOR_FLAG, 50, 3)
                        # sfx: ball_sink.wav
                        self.current_hole_index += 1
                        if self.current_hole_index >= self.NUM_HOLES:
                            self.game_over = True
                        else:
                            self._load_hole(self.current_hole_index)
                        break # Exit physics loop
                    physics_steps += 1
                
                # After physics loop
                if not self.game_over and self.game_state == 'BALL_MOVING': # if not sunk or penalized
                    self.game_state = 'AIMING'
                    current_dist_to_hole = self.ball_pos.distance_to(pygame.Vector2(self.current_course["hole"]))
                    if current_dist_to_hole < self.last_dist_to_hole:
                        reward += 0.5 # Progress reward
                    self.last_dist_to_hole = current_dist_to_hole

        # Termination checks
        if self.stroke_count >= self.MAX_STROKES:
            self.game_over = True
            reward -= 20 # Penalty for running out of strokes

        if self.game_over:
            terminated = True
            if self.current_hole_index >= self.NUM_HOLES:
                reward += 100 # Bonus for finishing
                if self.stroke_count <= 7:
                    reward += 50 # Extra bonus for being under par
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        screen_x = self.SCREEN_WIDTH / 2 + (x - y) * 0.7
        screen_y = self.SCREEN_HEIGHT * 0.2 + (x + y) * 0.35
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw course boundaries
        course = self.current_course
        course_points = [
            self._iso_to_screen(p[0], p[1]) for p in 
            [(0,0), (self.SCREEN_WIDTH, 0), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT/2), (0, self.SCREEN_HEIGHT/2)]
        ]
        
        # Draw grass tiles
        tile_size = 40
        for x in range(0, self.SCREEN_WIDTH, tile_size):
            for y in range(0, int(self.SCREEN_HEIGHT / 2), tile_size):
                color = self.COLOR_GRASS if (x // tile_size + y // tile_size) % 2 == 0 else self.COLOR_GRASS_DARK
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + tile_size, y)
                p3 = self._iso_to_screen(x + tile_size, y + tile_size)
                p4 = self._iso_to_screen(x, y + tile_size)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color)

        # Draw hole
        hole_pos_iso = self._iso_to_screen(*course["hole"])
        hole_radius_iso = int(course["radius"] * 0.8)
        pygame.gfxdraw.filled_ellipse(self.screen, hole_pos_iso[0], hole_pos_iso[1], hole_radius_iso, int(hole_radius_iso * 0.5), self.COLOR_HOLE)
        
        # Draw flag
        pole_top = (hole_pos_iso[0], hole_pos_iso[1] - 30)
        pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, hole_pos_iso, pole_top, 2)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [pole_top, (pole_top[0] + 15, pole_top[1] + 5), (pole_top[0], pole_top[1] + 10)])

        # Draw walls
        wall_height = 15
        for wall_start, wall_end in course["walls"]:
            p1 = self._iso_to_screen(*wall_start)
            p2 = self._iso_to_screen(*wall_end)
            p3 = (p2[0], p2[1] + wall_height)
            p4 = (p1[0], p1[1] + wall_height)
            
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_WALL)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_WALL)
            pygame.draw.line(self.screen, self.COLOR_WALL_TOP, p1, p2, 2)

        # Draw ball shadow
        ball_pos_iso = self._iso_to_screen(self.ball_pos.x, self.ball_pos.y)
        shadow_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.gfxdraw.filled_ellipse(shadow_surface, 10, 10, 8, 4, self.COLOR_SHADOW)
        self.screen.blit(shadow_surface, (ball_pos_iso[0] - 10, ball_pos_iso[1] - 4))

        # Draw ball
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_iso[0], ball_pos_iso[1] - 3, 6, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_iso[0], ball_pos_iso[1] - 3, 6, self.COLOR_BALL)

        # Draw aiming line
        if self.game_state == 'AIMING':
            line_len = self.shot_power * 1.5
            end_x = self.ball_pos.x + line_len * math.cos(self.aim_angle)
            end_y = self.ball_pos.y + line_len * math.sin(self.aim_angle)
            end_pos_iso = self._iso_to_screen(end_x, end_y)
            start_pos_iso = self._iso_to_screen(self.ball_pos.x, self.ball_pos.y)
            self._draw_dashed_line(start_pos_iso, end_pos_iso, self.COLOR_AIM)
    
    def _draw_dashed_line(self, start_pos, end_pos, color, dash_length=5):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        if distance == 0: return
        dashes = int(distance / dash_length / 2)
        
        for i in range(dashes):
            start = i * 2 * dash_length
            end = start + dash_length
            pos1 = (x1 + dx * start / distance, y1 + dy * start / distance)
            pos2 = (x1 + dx * end / distance, y1 + dy * end / distance)
            pygame.draw.line(self.screen, color, pos1, pos2, 2)

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p_iso = self._iso_to_screen(p['pos'].x, p['pos'].y)
                alpha = max(0, min(255, int(p['life'] * p['alpha_decay'])))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (p_iso[0] - p['size'], p_iso[1] - p['size']))

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': random.randint(15, 30),
                'color': color,
                'size': random.randint(2, 4),
                'alpha_decay': 255 / 30
            })

    def _render_ui(self):
        # Power Bar
        if self.game_state == 'AIMING':
            bar_width = 150
            bar_height = 20
            power_ratio = (self.shot_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
            
            bg_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - bar_width / 2 - 2, self.SCREEN_HEIGHT - 40 - 2, bar_width + 4, bar_height + 4)
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=5)
            
            fill_width = int(bar_width * power_ratio)
            fill_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - bar_width / 2, self.SCREEN_HEIGHT - 40, fill_width, bar_height)
            
            # Gradient for power bar
            color1 = (0, 255, 0)
            color2 = (255, 255, 0)
            color3 = (255, 0, 0)
            final_color = color1
            if power_ratio > 0.5:
                lerp_ratio = (power_ratio - 0.5) * 2
                final_color = tuple(int(c1 * (1 - lerp_ratio) + c2 * lerp_ratio) for c1, c2 in zip(color2, color3))
            else:
                lerp_ratio = power_ratio * 2
                final_color = tuple(int(c1 * (1 - lerp_ratio) + c2 * lerp_ratio) for c1, c2 in zip(color1, color2))
            
            pygame.draw.rect(self.screen, final_color, fill_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH / 2 - bar_width / 2, self.SCREEN_HEIGHT - 40, bar_width, bar_height), 1, border_radius=3)

        # Text UI
        stroke_text = self.font_large.render(f"Stroke: {self.stroke_count}/{self.MAX_STROKES}", True, self.COLOR_UI_TEXT)
        hole_text = self.font_large.render(f"Hole: {self.current_hole_index + 1}/{self.NUM_HOLES}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(stroke_text, (20, 10))
        self.screen.blit(hole_text, (self.SCREEN_WIDTH - hole_text.get_width() - 20, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_GRASS)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hole": self.current_hole_index + 1,
            "strokes": self.stroke_count,
        }

    def close(self):
        pygame.quit()

    def _collide_line_circle(self, p1, p2, circle_pos, circle_r, circle_vel):
        p1 = pygame.Vector2(p1)
        p2 = pygame.Vector2(p2)
        
        line_vec = p2 - p1
        line_mag_sq = line_vec.magnitude_squared()
        if line_mag_sq == 0: return False, circle_pos, circle_vel

        t = ((circle_pos - p1).dot(line_vec)) / line_mag_sq

        if 0 <= t <= 1:
            closest_point = p1 + t * line_vec
        elif t < 0:
            closest_point = p1
        else:
            closest_point = p2
        
        dist_vec = circle_pos - closest_point
        if dist_vec.magnitude() < circle_r:
            # Collision occurred
            overlap = circle_r - dist_vec.magnitude()
            correction = dist_vec.normalize() * overlap
            new_pos = circle_pos + correction
            
            if 0 <= t <= 1: # Perpendicular collision
                normal = line_vec.rotate(90).normalize()
            else: # Endpoint collision
                normal = dist_vec.normalize()
            
            new_vel = circle_vel.reflect(normal)
            return True, new_pos, new_vel

        return False, circle_pos, circle_vel

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Mini-Golf")
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to actions
            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2 # Not used in this mapping, but available
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]
        
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {info['steps']}, Stroke: {info['strokes']}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()