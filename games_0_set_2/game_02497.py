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
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to hit the ball. Hold shift to reset aim."
    )

    game_description = (
        "An isometric mini-golf game. Sink the ball in 5 holes with the fewest strokes. Aim for under par (15)!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_MARGIN = 50

        # Game constants
        self.PAR = 15
        self.MAX_STROKES = 25
        self.NUM_HOLES = 5
        self.FRICTION = 0.985
        self.MIN_VELOCITY_FOR_MOVE = 0.05
        self.MAX_POWER = 15
        self.BALL_RADIUS = 6
        self.HOLE_RADIUS = 10
        self.OBSTACLE_HIT_PENALTY_FACTOR = 0.7 # a bounce will reduce velocity

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRASS = (65, 152, 10)
        self.COLOR_ROUGH = (50, 120, 10)
        self.COLOR_WALL = (139, 125, 107)
        self.COLOR_WALL_TOP = (160, 145, 127)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_HOLE = (10, 10, 10)
        self.COLOR_FLAG_POLE = (200, 200, 200)
        self.COLOR_FLAG = (220, 40, 40)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_AIM_LINE = (255, 255, 255, 150)
        self.COLOR_POWER_LOW = (0, 255, 0)
        self.COLOR_POWER_HIGH = (255, 0, 0)

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
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 48, bold=True)

        # Game state variables
        self.game_state = "AIMING"  # AIMING, BALL_MOVING, HOLE_COMPLETE, GAME_OVER
        self.current_hole_index = 0
        self.total_strokes = 0
        self.hole_strokes = 0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.aim_angle = 0.0
        self.shot_power = self.MAX_POWER / 2
        self.last_space_held = False
        self.last_ball_dist_to_hole = 0
        self.win_state = False
        self.game_over = False
        self.steps = 0
        self.windmill_angle = 0
        self.particles = []

        self._define_holes()
        self.reset()
        
        # self.validate_implementation() # Call for self-check

    def _define_holes(self):
        self.holes = [
            # Hole 1: Straight shot
            {
                "start": pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 80),
                "hole": pygame.Vector2(self.WIDTH / 2, 80),
                "walls": [],
                "windmill": None,
            },
            # Hole 2: One wall
            {
                "start": pygame.Vector2(120, self.HEIGHT / 2),
                "hole": pygame.Vector2(self.WIDTH - 120, self.HEIGHT / 2),
                "walls": [pygame.Rect(self.WIDTH / 2 - 10, self.HEIGHT / 2 - 80, 20, 160)],
                "windmill": None,
            },
            # Hole 3: Dogleg right
            {
                "start": pygame.Vector2(120, self.HEIGHT - 80),
                "hole": pygame.Vector2(self.WIDTH - 120, 80),
                "walls": [
                    pygame.Rect(self.WIDTH / 2, 0, 20, 200),
                    pygame.Rect(self.WIDTH / 2, self.HEIGHT - 200, 20, 200),
                ],
                "windmill": None,
            },
            # Hole 4: Windmill
            {
                "start": pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 80),
                "hole": pygame.Vector2(self.WIDTH / 2, 80),
                "walls": [],
                "windmill": {"pos": pygame.Vector2(self.WIDTH/2, self.HEIGHT/2), "blade_len": 60, "blade_width": 8},
            },
            # Hole 5: Complex
            {
                "start": pygame.Vector2(100, self.HEIGHT - 80),
                "hole": pygame.Vector2(self.WIDTH - 100, 80),
                "walls": [
                    pygame.Rect(200, self.HEIGHT / 2 + 30, 100, 20),
                    pygame.Rect(self.WIDTH - 300, self.HEIGHT / 2 - 50, 100, 20),
                ],
                "windmill": None,
            },
        ]

    def _setup_hole(self, hole_index):
        self.current_hole_index = hole_index
        hole_data = self.holes[hole_index]
        self.ball_pos = hole_data["start"].copy()
        self.ball_vel.update(0, 0)
        self.hole_strokes = 0
        self.game_state = "AIMING"
        # Default aim towards the hole
        aim_vec = hole_data["hole"] - hole_data["start"]
        if aim_vec.length_squared() > 0:
            self.aim_angle = aim_vec.angle_to(pygame.Vector2(1, 0))
        else:
            self.aim_angle = 0.0
        self.last_ball_dist_to_hole = self.ball_pos.distance_to(hole_data["hole"])
        self.particles.clear()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_strokes = 0
        self.game_over = False
        self.win_state = False
        self.steps = 0
        self.last_space_held = False
        self._setup_hole(0)
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        reward = 0
        self.steps += 1
        
        # Update windmill regardless of state
        self.windmill_angle = (self.windmill_angle + 2) % 360

        if self.game_state == "AIMING":
            reward -= 0.01  # Small penalty for taking time to aim
            if movement == 1:  # Up
                self.shot_power = min(self.MAX_POWER, self.shot_power + 0.25)
            elif movement == 2:  # Down
                self.shot_power = max(1, self.shot_power - 0.25)
            elif movement == 3:  # Left
                self.aim_angle = (self.aim_angle - 2) % 360
            elif movement == 4:  # Right
                self.aim_angle = (self.aim_angle + 2) % 360
            
            if shift_held:
                hole_data = self.holes[self.current_hole_index]
                aim_vec = hole_data["hole"] - self.ball_pos
                if aim_vec.length_squared() > 0:
                    self.aim_angle = aim_vec.angle_to(pygame.Vector2(1, 0))

            if space_pressed:
                # --- HIT THE BALL ---
                self.game_state = "BALL_MOVING"
                self.hole_strokes += 1
                self.total_strokes += 1
                self.ball_vel = pygame.Vector2(1, 0).rotate(-self.aim_angle) * (self.shot_power)
                self.last_ball_dist_to_hole = self.ball_pos.distance_to(self.holes[self.current_hole_index]["hole"])
                
                # --- Physics Simulation Loop ---
                sub_steps = 0
                hit_obstacle_this_turn = False
                
                while self.ball_vel.length() > self.MIN_VELOCITY_FOR_MOVE and sub_steps < 300:
                    self.ball_pos += self.ball_vel
                    self.ball_vel *= self.FRICTION
                    
                    # Boundary checks
                    if not (self.BALL_RADIUS < self.ball_pos.x < self.WIDTH - self.BALL_RADIUS and \
                            self.BALL_RADIUS < self.ball_pos.y < self.HEIGHT - self.BALL_RADIUS):
                        reward -= 5 # Out of bounds penalty
                        self.hole_strokes += 1 # Penalty stroke
                        self.total_strokes += 1
                        self.ball_pos = self.holes[self.current_hole_index]["start"].copy()
                        self.ball_vel.update(0,0)
                        break

                    # Obstacle collision
                    current_hole = self.holes[self.current_hole_index]
                    for wall in current_hole["walls"]:
                        if wall.collidepoint(self.ball_pos):
                            # Find closest point on rect to ball, reflect velocity based on that normal
                            closest_x = max(wall.left, min(self.ball_pos.x, wall.right))
                            closest_y = max(wall.top, min(self.ball_pos.y, wall.bottom))
                            
                            # The vector from the closest point on the wall to the ball's center.
                            # This vector is (0,0) if the ball's center is inside the wall,
                            # which would cause a crash on .normalize().
                            collision_vec = self.ball_pos - pygame.Vector2(closest_x, closest_y)
                            
                            if collision_vec.length_squared() > 1e-6:
                                normal = collision_vec.normalize()
                            else:
                                # The ball center is inside the wall. This is a logic error in the original
                                # collision detection. We recover by creating a fallback normal.
                                # We'll use the inverse of the velocity direction.
                                if self.ball_vel.length_squared() > 1e-6:
                                    normal = -self.ball_vel.normalize()
                                else:
                                    # Velocity is also zero, ball is stuck. Push it out in an arbitrary stable direction.
                                    normal = pygame.Vector2(0, -1)

                            self.ball_vel = self.ball_vel.reflect(normal) * self.OBSTACLE_HIT_PENALTY_FACTOR
                            self.ball_pos += normal # Push ball out of wall to prevent getting stuck
                            if not hit_obstacle_this_turn:
                                reward -= 2 # Penalty for hitting obstacle
                                hit_obstacle_this_turn = True
                    
                    # Windmill collision
                    if current_hole["windmill"]:
                        wm = current_hole["windmill"]
                        for i in range(2): # Two blades
                            angle = self.windmill_angle + i * 180
                            rad_angle = math.radians(angle)
                            p1 = wm["pos"]
                            p2 = wm["pos"] + pygame.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * wm["blade_len"]
                            
                            blade_rect = pygame.Rect(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y)
                            blade_rect.normalize()
                            blade_rect.inflate_ip(wm["blade_width"], wm["blade_width"])

                            if blade_rect.collidepoint(self.ball_pos):
                                normal_vec = (p2-p1).rotate(90)
                                if normal_vec.length_squared() > 0:
                                    normal = normal_vec.normalize()
                                    self.ball_vel = self.ball_vel.reflect(normal) * self.OBSTACLE_HIT_PENALTY_FACTOR
                                    self.ball_pos += normal
                                    if not hit_obstacle_this_turn:
                                        reward -= 2
                                        hit_obstacle_this_turn = True

                    # Check for sinking the ball
                    hole_pos = current_hole["hole"]
                    if self.ball_pos.distance_to(hole_pos) < self.HOLE_RADIUS:
                        self.ball_vel.update(0, 0)
                        self.ball_pos = hole_pos.copy()
                        self.game_state = "HOLE_COMPLETE"
                        reward += 10 # Reward for sinking
                        for _ in range(50):
                            self.particles.append(Particle(hole_pos))
                        break

                    sub_steps += 1
                
                # After ball stops moving
                if self.game_state != "HOLE_COMPLETE":
                    self.game_state = "AIMING"
                    new_dist = self.ball_pos.distance_to(self.holes[self.current_hole_index]["hole"])
                    dist_improvement = (self.last_ball_dist_to_hole - new_dist) / self.WIDTH
                    reward += dist_improvement * 5
                    self.last_ball_dist_to_hole = new_dist
        
        # Check termination conditions
        terminated = False
        if self.game_state == "HOLE_COMPLETE":
            if self.current_hole_index + 1 >= self.NUM_HOLES:
                self.game_over = True
                terminated = True
                self.win_state = self.total_strokes <= self.PAR
                if self.win_state:
                    reward += 50 # Under par bonus
                else:
                    reward += 25 # Finish bonus
            else:
                self._setup_hole(self.current_hole_index + 1)
        
        if self.total_strokes >= self.MAX_STROKES:
            self.game_over = True
            terminated = True
            self.win_state = False
            reward -= 100 # Penalty for maxing out strokes

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_course()
        self._render_ball_and_hole()
        if self.game_state == "AIMING":
            self._render_aim_assists()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.total_strokes,
            "steps": self.steps,
            "current_hole": self.current_hole_index + 1,
            "hole_strokes": self.hole_strokes,
            "game_over": self.game_over,
            "win_state": self.win_state
        }
    
    def _render_course(self):
        # Draw main grass area
        pygame.draw.rect(self.screen, self.COLOR_GRASS, (self.WORLD_MARGIN, self.WORLD_MARGIN, 
                                                         self.WIDTH - 2*self.WORLD_MARGIN, self.HEIGHT - 2*self.WORLD_MARGIN),
                                                         border_radius=10)
        # Draw rough border
        pygame.draw.rect(self.screen, self.COLOR_ROUGH, (self.WORLD_MARGIN-10, self.WORLD_MARGIN-10, 
                                                          self.WIDTH - 2*self.WORLD_MARGIN + 20, self.HEIGHT - 2*self.WORLD_MARGIN + 20),
                                                          width=10, border_radius=15)
        
        # Draw obstacles for the current hole
        hole_data = self.holes[self.current_hole_index]
        for wall in hole_data["walls"]:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            top_face = wall.copy()
            top_face.height = max(5, wall.height * 0.2)
            pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, top_face)
        
        # Draw windmill
        if hole_data["windmill"]:
            wm = hole_data["windmill"]
            pygame.gfxdraw.filled_circle(self.screen, int(wm["pos"].x), int(wm["pos"].y), 10, self.COLOR_WALL)
            pygame.gfxdraw.aacircle(self.screen, int(wm["pos"].x), int(wm["pos"].y), 10, self.COLOR_WALL_TOP)
            for i in range(2):
                angle = self.windmill_angle + i * 180
                rad_angle = math.radians(angle)
                end_pos = wm["pos"] + pygame.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * wm["blade_len"]
                pygame.draw.line(self.screen, self.COLOR_WALL_TOP, wm["pos"], end_pos, wm["blade_width"])


    def _render_ball_and_hole(self):
        hole_pos = self.holes[self.current_hole_index]["hole"]
        
        # Hole
        pygame.gfxdraw.filled_circle(self.screen, int(hole_pos.x), int(hole_pos.y), self.HOLE_RADIUS, self.COLOR_HOLE)
        
        # Flag
        if self.game_state != "HOLE_COMPLETE":
            flag_pole_start = (int(hole_pos.x), int(hole_pos.y))
            flag_pole_end = (int(hole_pos.x), int(hole_pos.y) - 25)
            pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, flag_pole_start, flag_pole_end, 2)
            flag_points = [flag_pole_end, (int(hole_pos.x) - 12, int(hole_pos.y) - 20), (int(hole_pos.x), int(hole_pos.y) - 15)]
            pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
            pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
            
        # Ball shadow
        shadow_pos = (int(self.ball_pos.x) + 3, int(self.ball_pos.y) + 3)
        pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], self.BALL_RADIUS, self.BALL_RADIUS-2, (0,0,0,100))
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, (200,200,200))

    def _render_aim_assists(self):
        # Aim line
        angle_rad = -math.radians(self.aim_angle)
        line_len = 20 + self.shot_power * 5
        end_pos = self.ball_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * line_len
        
        # Dashed line
        num_dashes = 10
        for i in range(num_dashes):
            if i % 2 == 0:
                start = self.ball_pos.lerp(end_pos, i / num_dashes)
                end = self.ball_pos.lerp(end_pos, (i+1) / num_dashes)
                pygame.draw.line(self.screen, self.COLOR_AIM_LINE, start, end, 2)

        # Power bar
        power_percent = (self.shot_power - 1) / (self.MAX_POWER - 1)
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH / 2 - bar_width / 2
        bar_y = self.HEIGHT - 35
        
        # Interpolate color for power bar
        power_color = (
            self.COLOR_POWER_LOW[0] * (1 - power_percent) + self.COLOR_POWER_HIGH[0] * power_percent,
            self.COLOR_POWER_LOW[1] * (1 - power_percent) + self.COLOR_POWER_HIGH[1] * power_percent,
            self.COLOR_POWER_LOW[2] * (1 - power_percent) + self.COLOR_POWER_HIGH[2] * power_percent,
        )

        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, power_color, (bar_x, bar_y, bar_width * power_percent, bar_height), border_radius=5)
        power_text = self.font_small.render("POWER", True, self.COLOR_TEXT)
        self.screen.blit(power_text, (bar_x + bar_width/2 - power_text.get_width()/2, bar_y + bar_height/2 - power_text.get_height()/2))

    def _render_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)
            else:
                p.draw(self.screen)

    def _render_ui(self):
        # Hole info
        hole_text = self.font_large.render(f"HOLE {self.current_hole_index + 1}", True, self.COLOR_TEXT)
        self.screen.blit(hole_text, (20, 10))
        
        # Strokes info
        strokes_text = self.font_small.render(f"STROKES: {self.hole_strokes}", True, self.COLOR_TEXT)
        self.screen.blit(strokes_text, (20, 50))
        
        # Total strokes and Par
        total_text = self.font_large.render(f"TOTAL: {self.total_strokes}", True, self.COLOR_TEXT)
        self.screen.blit(total_text, (self.WIDTH - total_text.get_width() - 20, 10))
        par_text = self.font_small.render(f"PAR: {self.PAR}", True, self.COLOR_TEXT)
        self.screen.blit(par_text, (self.WIDTH - par_text.get_width() - 20, 50))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win_state:
            msg = "COURSE COMPLETE!"
            color = (100, 255, 100)
        else:
            msg = "GAME OVER"
            color = (255, 100, 100)

        title_text = self.font_huge.render(msg, True, color)
        self.screen.blit(title_text, (self.WIDTH/2 - title_text.get_width()/2, self.HEIGHT/2 - 50))
        
        final_score_text = self.font_large.render(f"Final Score: {self.total_strokes}", True, self.COLOR_TEXT)
        self.screen.blit(final_score_text, (self.WIDTH/2 - final_score_text.get_width()/2, self.HEIGHT/2 + 20))
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

class Particle:
    def __init__(self, pos):
        self.pos = pos.copy()
        angle = random.uniform(0, 360)
        speed = random.uniform(1, 4)
        self.vel = pygame.Vector2(speed, 0).rotate(angle)
        self.lifespan = random.randint(20, 40)
        self.color = random.choice([(255, 255, 0), (255, 200, 0), (255, 255, 255)])
        self.radius = random.randint(2, 4)

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.95

    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, surface):
        alpha = max(0, 255 * (self.lifespan / 20))
        color = (*self.color, alpha)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, color)