
# Generated: 2025-08-27T22:23:53.613985
# Source Brief: brief_03110.md
# Brief Index: 3110

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to swing. Shift resets aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A 2D isometric mini-golf game. Sink the ball in the fewest strokes across 9 procedurally generated holes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_FAIRWAY = (70, 160, 80)
        self.COLOR_ROUGH = (50, 120, 60)
        self.COLOR_WATER = (60, 100, 200)
        self.COLOR_WALL = (100, 100, 110)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_HOLE = (10, 10, 10)
        self.COLOR_AIM = (255, 255, 255, 150)
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_POWER_BAR_BG = (80, 80, 80)
        self.COLOR_POWER_BAR_FILL = (255, 200, 0)
        
        # Game constants
        self.WORLD_WIDTH = 30
        self.WORLD_HEIGHT = 20
        self.TILE_WIDTH_HALF = 16
        self.TILE_HEIGHT_HALF = 8
        self.MAX_STROKES_PER_HOLE = 10
        self.MAX_EPISODE_STEPS = 1000
        self.BALL_RADIUS = 5
        self.HOLE_RADIUS = 7
        self.FRICTION_FAIRWAY = 0.98
        self.FRICTION_ROUGH = 0.95
        self.MIN_VELOCITY = 0.05
        
        # State variables will be initialized in reset()
        self.game_state = None
        self.current_hole_num = None
        self.strokes_current_hole = None
        self.total_strokes = None
        self.ball_pos = None
        self.ball_vel = None
        self.hole_pos = None
        self.start_pos = None
        self.course_layout = None
        self.obstacles = None
        self.aim_angle = None
        self.aim_power = None
        self.particles = None
        self.game_over_message = None
        self.steps = 0
        self.last_dist_to_hole = float('inf')
        self.score = 0 # Total strokes, as per info dict requirement

        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_strokes = 0
        self.score = 0
        self.current_hole_num = 1
        self.game_over_message = None
        self.particles = []
        
        self._generate_new_hole()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_new_hole(self):
        self.game_state = "AIMING"
        self.strokes_current_hole = 0
        self.last_dist_to_hole = float('inf')

        # Generate a simple course layout
        self.course_layout = [[0 for _ in range(self.WORLD_WIDTH)] for _ in range(self.WORLD_HEIGHT)]
        
        start_y = self.np_random.integers(5, self.WORLD_HEIGHT - 5)
        self.start_pos = pygame.Vector2(self.np_random.integers(2, 5), start_y)
        
        hole_y = self.np_random.integers(5, self.WORLD_HEIGHT - 5)
        self.hole_pos = pygame.Vector2(self.np_random.integers(self.WORLD_WIDTH - 5, self.WORLD_WIDTH - 2), hole_y)
        
        self.ball_pos = self.start_pos.copy()
        self.ball_vel = pygame.Vector2(0, 0)

        # Create a path of fairway and rough
        path_points = [self.start_pos, self.hole_pos]
        num_mid_points = self.current_hole_num // 2
        for _ in range(num_mid_points):
            mid_x = self.np_random.uniform(self.start_pos.x, self.hole_pos.x)
            mid_y = self.np_random.uniform(0, self.WORLD_HEIGHT)
            path_points.insert(1, pygame.Vector2(mid_x, mid_y))
        
        for i in range(len(path_points) - 1):
            self._carve_path(path_points[i], path_points[i+1], 1, 4) # Rough
            self._carve_path(path_points[i], path_points[i+1], 2, 2) # Fairway
        
        # Place wall obstacles
        self.obstacles = []
        num_obstacles = min(self.current_hole_num, 5)
        for _ in range(num_obstacles):
            for _ in range(10): # Max attempts to place an obstacle
                obs_pos = pygame.Vector2(
                    self.np_random.integers(0, self.WORLD_WIDTH),
                    self.np_random.integers(0, self.WORLD_HEIGHT)
                )
                if obs_pos.distance_to(self.start_pos) > 5 and obs_pos.distance_to(self.hole_pos) > 5 and self.course_layout[int(obs_pos.y)][int(obs_pos.x)] != 0:
                    self.obstacles.append(obs_pos)
                    self.course_layout[int(obs_pos.y)][int(obs_pos.x)] = 3 # Wall
                    break
        
        self.aim_angle = math.atan2(self.hole_pos.y - self.ball_pos.y, self.hole_pos.x - self.ball_pos.x)
        self.aim_power = 0.5

    def _carve_path(self, p1, p2, tile_type, width=2):
        dist = p1.distance_to(p2)
        if dist == 0: return
        for i in range(int(dist * 2)):
            t = i / (dist * 2)
            current_pos = p1.lerp(p2, t)
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    if dx*dx + dy*dy <= width*width:
                        px, py = int(current_pos.x + dx), int(current_pos.y + dy)
                        if 0 <= px < self.WORLD_WIDTH and 0 <= py < self.WORLD_HEIGHT:
                            if self.course_layout[py][px] < tile_type:
                                self.course_layout[py][px] = tile_type

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = -0.1  # Cost of existing
        terminated = False

        if self.game_state == "AIMING":
            if movement == 1: self.aim_power = min(1.0, self.aim_power + 0.05)
            elif movement == 2: self.aim_power = max(0.0, self.aim_power - 0.05)
            elif movement == 3: self.aim_angle -= 0.08
            elif movement == 4: self.aim_angle += 0.08
            
            if shift_held:
                self.aim_angle = math.atan2(self.hole_pos.y - self.ball_pos.y, self.hole_pos.x - self.ball_pos.x)

            if space_held:
                # sfx: golf_swing.wav
                self.strokes_current_hole += 1
                self.total_strokes += 1
                self.score = self.total_strokes
                power = self.aim_power * 1.5
                self.ball_vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * power
                self.game_state = "SIMULATING"
                self.last_dist_to_hole = self.ball_pos.distance_to(self.hole_pos)
        
        elif self.game_state == "SIMULATING":
            reward += self._update_physics()
            
        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.game_over_message is None: # Terminated by step limit
                self.game_over_message = "TIME LIMIT REACHED"
            reward += self._calculate_terminal_reward()
            self.game_state = "GAME_OVER"
            
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_physics(self):
        reward = 0
        self.ball_pos += self.ball_vel
        
        current_dist = self.ball_pos.distance_to(self.hole_pos)
        if current_dist < self.HOLE_RADIUS / self.TILE_WIDTH_HALF: # Close enough to hole
            reward += 0.5
        
        ball_tile_x, ball_tile_y = int(self.ball_pos.x), int(self.ball_pos.y)
        if 0 <= ball_tile_x < self.WORLD_WIDTH and 0 <= ball_tile_y < self.WORLD_HEIGHT:
            tile_type = self.course_layout[ball_tile_y][ball_tile_x]
            if tile_type == 1: self.ball_vel *= self.FRICTION_ROUGH
            else: self.ball_vel *= self.FRICTION_FAIRWAY
        else: self.ball_vel *= self.FRICTION_FAIRWAY

        bx, by = self.ball_pos.x, self.ball_pos.y
        if not (0 <= bx < self.WORLD_WIDTH and 0 <= by < self.WORLD_HEIGHT) or (0 <= int(by) < self.WORLD_HEIGHT and 0 <= int(bx) < self.WORLD_WIDTH and self.course_layout[int(by)][int(bx)] == 0):
            # sfx: water_splash.wav
            self.game_over_message = "OUT OF BOUNDS!"
            self.ball_vel = pygame.Vector2(0, 0)
        elif self.course_layout[int(by)][int(bx)] == 3: # Wall
            # sfx: ball_hit_wall.wav
            self._create_particles(self.ball_pos, self.COLOR_WALL, 5)
            tile_center = pygame.Vector2(int(bx) + 0.5, int(by) + 0.5)
            diff = self.ball_pos - tile_center
            if abs(diff.x) > abs(diff.y): self.ball_vel.x *= -0.8
            else: self.ball_vel.y *= -0.8
            self.ball_pos += self.ball_vel * 0.1 # Push out of wall
        
        if self.ball_vel.magnitude() < self.MIN_VELOCITY:
            self.ball_vel = pygame.Vector2(0, 0)
            if self.ball_pos.distance_to(self.hole_pos) < self.HOLE_RADIUS / self.TILE_WIDTH_HALF:
                # sfx: hole_sink.wav
                self._create_particles(self.hole_pos, self.COLOR_BALL, 50)
                if self.current_hole_num >= 9:
                    self.game_over_message = f"COURSE COMPLETE! Total: {self.total_strokes}"
                else:
                    self.current_hole_num += 1
                    self._generate_new_hole()
            else:
                self.game_state = "AIMING"
        return reward
    
    def _check_termination(self):
        if self.game_over_message is not None:
            return True
        if self.strokes_current_hole >= self.MAX_STROKES_PER_HOLE:
            self.game_over_message = f"MAX STROKES ({self.MAX_STROKES_PER_HOLE}) REACHED!"
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
        return False

    def _calculate_terminal_reward(self):
        if "COMPLETE" in (self.game_over_message or ""): return 50
        if "OUT OF BOUNDS" in (self.game_over_message or ""): return -5
        if "MAX STROKES" in (self.game_over_message or ""): return -2
        return 0

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
            "hole": self.current_hole_num,
            "hole_strokes": self.strokes_current_hole,
        }

    def _world_to_screen(self, x, y):
        screen_x = self.screen_width // 2 + (x - y) * self.TILE_WIDTH_HALF
        screen_y = 50 + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Render course tiles
        for y in range(self.WORLD_HEIGHT):
            for x in range(self.WORLD_WIDTH):
                tile_type = self.course_layout[y][x]
                color = self.COLOR_WATER
                if tile_type == 1: color = self.COLOR_ROUGH
                elif tile_type == 2: color = self.COLOR_FAIRWAY
                elif tile_type == 3: color = self.COLOR_WALL
                
                sx, sy = self._world_to_screen(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF), (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF), (sx - self.TILE_WIDTH_HALF, sy)
                ]
                pygame.draw.polygon(self.screen, color, points)
        
        hole_sx, hole_sy = self._world_to_screen(self.hole_pos.x, self.hole_pos.y)
        pygame.gfxdraw.filled_ellipse(self.screen, hole_sx, hole_sy, self.HOLE_RADIUS, self.HOLE_RADIUS // 2, self.COLOR_HOLE)
        pygame.gfxdraw.aaellipse(self.screen, hole_sx, hole_sy, self.HOLE_RADIUS, self.HOLE_RADIUS // 2, self.COLOR_HOLE)
        
        ball_sx, ball_sy = self._world_to_screen(self.ball_pos.x, self.ball_pos.y)
        pygame.gfxdraw.filled_ellipse(self.screen, ball_sx, ball_sy + 2, self.BALL_RADIUS, self.BALL_RADIUS // 2, (0,0,0,100))
        pygame.gfxdraw.filled_circle(self.screen, ball_sx, ball_sy, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_sx, ball_sy, self.BALL_RADIUS, self.COLOR_BALL)

        if self.game_state == "AIMING":
            sim_pos, power = self.ball_pos.copy(), self.aim_power * 1.5
            sim_vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * power
            points = []
            for _ in range(20):
                sim_pos += sim_vel * 0.2
                sim_vel *= self.FRICTION_FAIRWAY**2
                sx, sy = self._world_to_screen(sim_pos.x, sim_pos.y)
                points.append((sx, sy))
            if len(points) > 1: pygame.draw.aalines(self.screen, self.COLOR_AIM, False, points, 1)

        for p in self.particles:
            sx, sy = self._world_to_screen(p['pos'].x, p['pos'].y)
            alpha = max(0, min(255, int(p['life'] * 6)))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, (sx, sy), 2)
    
    def _render_ui(self):
        hole_text = self.font_small.render(f"Hole: {self.current_hole_num}/9", True, self.COLOR_UI)
        strokes_text = self.font_small.render(f"Strokes: {self.strokes_current_hole} ({self.total_strokes})", True, self.COLOR_UI)
        self.screen.blit(hole_text, (10, 10))
        self.screen.blit(strokes_text, (10, 30))

        if self.game_state == "AIMING":
            bar_width, bar_height = 150, 20
            bar_x, bar_y = self.screen_width - bar_width - 10, self.screen_height - bar_height - 10
            fill_width = bar_width * self.aim_power
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            if fill_width > 0: pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
            power_text = self.font_small.render("POWER", True, self.COLOR_UI)
            self.screen.blit(power_text, (bar_x, bar_y - 20))

        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_UI)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA); s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect); self.screen.blit(msg_surf, msg_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.05, 0.2)
            self.particles.append({
                'pos': pos.copy(), 'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(20, 40), 'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    import os
    import time

    # To play interactively, you need a display.
    # Set this to False to run the headless validation checks.
    INTERACTIVE_PLAY = True 
    
    if not INTERACTIVE_PLAY:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    if not INTERACTIVE_PLAY:
        print("Running headless validation...")
        env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: Reward={reward:.2f}, Terminated={terminated}, Info={info}")
            if terminated:
                print("Episode finished. Resetting.")
                env.reset()
        env.close()
        print("Headless validation complete.")
    else:
        print("\nStarting interactive play...")
        obs, info = env.reset()
        done = False
        display_screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Mini-Golf")
        clock = pygame.time.Clock()

        while not done:
            movement, space, shift = 0, 0, 0
            
            # This is a turn-based game, so we only need one action per frame
            # when the game is in the AIMING state.
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and env.game_state == "AIMING":
                    action_taken = True
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
            
            # If no key was pressed, we can send a no-op action
            # or just wait. Since auto_advance is False, we must send an action.
            if env.game_state == "SIMULATING" or not action_taken:
                action = [0, 0, 0] # no-op
            else:
                action = [movement, space, shift]

            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}. Resetting in 3 seconds.")
                time.sleep(3)
                obs, info = env.reset()
                
            clock.tick(30)
        
        env.close()