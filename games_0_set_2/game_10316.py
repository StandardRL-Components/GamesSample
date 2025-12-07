import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Define data structures for clarity
Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifespan"])
TunnelPoint = namedtuple("TunnelPoint", ["center_x", "width"])
Trigger = namedtuple("Trigger", ["world_y", "x_offset", "active"])

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a futuristic tunnel racing game.
    The player pilots a vehicle through a procedurally generated neon tunnel,
    avoiding walls and hitting triggers to score points against a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a futuristic ship through a procedurally generated neon tunnel. "
        "Avoid the walls and activate scoring triggers to maximize your score before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ to steer the ship. Press space to activate score triggers as you fly over them."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60

    # Colors
    COLOR_BG = (5, 0, 15)
    COLOR_TUNNEL_FILL = (10, 5, 40)
    COLOR_TUNNEL_EDGE = (50, 100, 255)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 30)
    COLOR_TRIGGER = (255, 150, 0)
    COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (200, 50, 0)]
    COLOR_UI_TEXT = (255, 255, 255)

    # Game parameters
    MAX_STEPS = 90 * TARGET_FPS  # 90 seconds
    WIN_SCORE = 2000
    FORWARD_SPEED = 5.0  # Pixels per frame the world scrolls
    PLAYER_Y_POSITION = 320  # Fixed Y position of the player on the screen

    # Player physics
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.92  # Damping factor for horizontal velocity
    PLAYER_MAX_VX = 8.0
    PLAYER_VISUAL_TILT_FACTOR = 3.0 # How much the ship tilts when moving

    # Tunnel generation
    TUNNEL_LENGTH = MAX_STEPS + SCREEN_HEIGHT // FORWARD_SPEED + 100
    TUNNEL_SEGMENT_LENGTH = 10
    TUNNEL_MIN_WIDTH = 150
    TUNNEL_MAX_WIDTH = 300

    # Rewards
    REWARD_SURVIVAL = 0.01  # Small reward for each step alive
    REWARD_TRIGGER = 50.0
    REWARD_WIN = 100.0
    REWARD_CRASH = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 30)

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_x = 0
        self.player_vx = 0
        self.player_angle = 0
        self.scroll_y = 0
        self.tunnel = []
        self.triggers = []
        self.particles = []
        
        # self.reset() is called by the test harness, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_x = self.SCREEN_WIDTH / 2
        self.player_vx = 0.0
        self.player_angle = 0.0

        # World state
        self.scroll_y = 0.0
        self._generate_tunnel()
        self._generate_triggers()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # According to Gymnasium spec, step should not be called after termination.
            # If it is, we return a valid 5-tuple consistent with a terminal state.
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- 1. HANDLE INPUT ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        player_ax = 0.0
        if movement == 3:  # Left
            player_ax = -self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            player_ax = self.PLAYER_ACCELERATION

        # --- 2. UPDATE GAME STATE ---
        # Update player physics
        self.player_vx += player_ax
        self.player_vx *= self.PLAYER_FRICTION
        self.player_vx = np.clip(self.player_vx, -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        self.player_x += self.player_vx
        self.player_x = np.clip(self.player_x, 0, self.SCREEN_WIDTH)
        self.player_angle = self.player_vx * self.PLAYER_VISUAL_TILT_FACTOR

        # Update world scroll
        self.scroll_y += self.FORWARD_SPEED

        # Update particles
        self._update_particles()
        
        # --- 3. CALCULATE REWARD & CHECK INTERACTIONS ---
        reward = self.REWARD_SURVIVAL
        
        # Check for trigger activation
        if space_held:
            reward += self._check_triggers()

        # --- 4. CHECK TERMINATION CONDITIONS ---
        terminated = False
        
        # Collision
        player_world_y = self.scroll_y + self.PLAYER_Y_POSITION
        left_bound, right_bound = self._get_tunnel_bounds_at(player_world_y)
        if not (left_bound < self.player_x < right_bound):
            terminated = True
            reward = self.REWARD_CRASH
            # SFX: Crash sound
            self._spawn_explosion((self.player_x, self.PLAYER_Y_POSITION), 30, 8.0)
        
        # Win condition
        if not terminated and self.score >= self.WIN_SCORE:
            terminated = True
            reward += self.REWARD_WIN
        
        # Timeout
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS - 1:
            truncated = True
        
        self.game_over = terminated or truncated
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "time_left": (self.MAX_STEPS - self.steps) / self.TARGET_FPS,
        }

    def close(self):
        pygame.quit()

    # --- HELPER & RENDERING METHODS ---

    def _generate_tunnel(self):
        self.tunnel = []
        # Use sum of sines for smooth, repeatable curves
        freq1, amp1 = self.np_random.uniform(0.002, 0.004), self.np_random.uniform(100, 150)
        freq2, amp2 = self.np_random.uniform(0.008, 0.012), self.np_random.uniform(40, 60)
        phase1, phase2 = self.np_random.uniform(0, 2*math.pi), self.np_random.uniform(0, 2*math.pi)

        # Add a straight, centered starting section for stability during no-op tests
        straight_start_segments = 150 

        for i in range(int(self.TUNNEL_LENGTH)):
            y = i * self.TUNNEL_SEGMENT_LENGTH
            
            if i < straight_start_segments:
                x_offset = 0.0
            else:
                x_offset = amp1 * math.sin(freq1 * y + phase1) + amp2 * math.sin(freq2 * y + phase2)
            
            center_x = self.SCREEN_WIDTH / 2 + x_offset
            
            # Width variation
            width_progress = min(1.0, i / 200) # Tunnel starts wide and narrows
            width = self.TUNNEL_MAX_WIDTH - (self.TUNNEL_MAX_WIDTH - self.TUNNEL_MIN_WIDTH) * width_progress
            
            self.tunnel.append(TunnelPoint(center_x, width))

    def _generate_triggers(self):
        self.triggers = []
        for i in range(50, int(self.TUNNEL_LENGTH) - 200, self.np_random.integers(15, 40)):
            world_y = i * self.TUNNEL_SEGMENT_LENGTH
            point_idx = i
            if point_idx < len(self.tunnel):
                tunnel_point = self.tunnel[point_idx]
                # Place trigger randomly within the tunnel width
                x_offset = self.np_random.uniform(-tunnel_point.width * 0.4, tunnel_point.width * 0.4)
                self.triggers.append(Trigger(world_y, x_offset, True))

    def _get_tunnel_bounds_at(self, world_y):
        idx_float = world_y / self.TUNNEL_SEGMENT_LENGTH
        idx0 = int(idx_float)
        
        if not (0 <= idx0 < len(self.tunnel) - 1):
            return 0, self.SCREEN_WIDTH # Out of tunnel, free space

        p1 = self.tunnel[idx0]
        p2 = self.tunnel[idx0 + 1]
        
        interp_factor = idx_float - idx0
        
        center_x = p1.center_x + (p2.center_x - p1.center_x) * interp_factor
        width = p1.width + (p2.width - p1.width) * interp_factor
        
        return center_x - width / 2, center_x + width / 2

    def _check_triggers(self):
        reward = 0
        player_world_y = self.scroll_y + self.PLAYER_Y_POSITION
        
        for i, trigger in enumerate(self.triggers):
            if trigger.active and abs(player_world_y - trigger.world_y) < 20:
                left_bound, right_bound = self._get_tunnel_bounds_at(trigger.world_y)
                trigger_x = (left_bound + right_bound) / 2 + trigger.x_offset
                if abs(self.player_x - trigger_x) < 25: # Activation proximity
                    self.triggers[i] = trigger._replace(active=False)
                    self.score += self.REWARD_TRIGGER
                    reward += self.REWARD_TRIGGER
                    # SFX: Trigger activation sound
                    screen_y = trigger.world_y - self.scroll_y
                    self._spawn_explosion((trigger_x, screen_y), 20, 5.0)
                    break # Only one trigger at a time
        return reward

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            new_pos = (p.pos[0] + p.vel[0], p.pos[1] + p.vel[1])
            new_lifespan = p.lifespan - 1
            if new_lifespan > 0:
                active_particles.append(p._replace(pos=new_pos, lifespan=new_lifespan))
        self.particles = active_particles

    def _spawn_explosion(self, pos, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = random.uniform(2, 6)
            color = random.choice(self.COLOR_EXPLOSION)
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _render_game(self):
        self._render_tunnel()
        self._render_triggers()
        self._render_particles()
        if not self.game_over:
            self._render_player()

    def _render_tunnel(self):
        start_idx = max(0, int(self.scroll_y / self.TUNNEL_SEGMENT_LENGTH))
        end_idx = min(len(self.tunnel), start_idx + int(self.SCREEN_HEIGHT / self.TUNNEL_SEGMENT_LENGTH) + 2)

        if end_idx <= start_idx + 1: return

        left_wall_pts = []
        right_wall_pts = []
        
        for i in range(start_idx, end_idx):
            point = self.tunnel[i]
            screen_y = i * self.TUNNEL_SEGMENT_LENGTH - self.scroll_y
            left_wall_pts.append((point.center_x - point.width / 2, screen_y))
            right_wall_pts.append((point.center_x + point.width / 2, screen_y))

        # Create a polygon for the tunnel floor
        tunnel_poly = left_wall_pts + right_wall_pts[::-1]
        pygame.draw.polygon(self.screen, self.COLOR_TUNNEL_FILL, tunnel_poly)
        
        # Draw neon edges with antialiasing
        pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_EDGE, False, left_wall_pts, 2)
        pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_EDGE, False, right_wall_pts, 2)

    def _render_triggers(self):
        for trigger in self.triggers:
            if not trigger.active: continue
            
            screen_y = trigger.world_y - self.scroll_y
            if 0 < screen_y < self.SCREEN_HEIGHT:
                left_bound, right_bound = self._get_tunnel_bounds_at(trigger.world_y)
                trigger_x = (left_bound + right_bound) / 2 + trigger.x_offset
                
                # Draw a glowing diamond shape for the trigger
                points = [
                    (trigger_x, screen_y - 10), (trigger_x + 10, screen_y),
                    (trigger_x, screen_y + 10), (trigger_x - 10, screen_y)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_TRIGGER, points)
                pygame.draw.aalines(self.screen, (255, 255, 255), True, points)


    def _render_player(self):
        # Base ship shape (triangle)
        ship_points = [(-8, 10), (0, -12), (8, 10)]
        
        # Rotate points based on player angle (tilt)
        angle_rad = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = []
        for x, y in ship_points:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_points.append((self.player_x + rx, self.PLAYER_Y_POSITION + ry))
        
        # Draw glow effect by drawing larger, transparent polygons
        for i in range(5, 0, -1):
            glow_scale = 1.0 + i * 0.3
            glow_points = []
            for x, y in ship_points:
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a
                glow_points.append((self.player_x + rx * glow_scale, self.PLAYER_Y_POSITION + ry * glow_scale))
            
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in glow_points], self.COLOR_PLAYER_GLOW)

        # Draw main ship body
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p.lifespan / 30.0) # Fade out
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.TARGET_FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Tunnel Runner")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        action = [movement, space, shift]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Resetting in 2 seconds...")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(GameEnv.TARGET_FPS)

    env.close()