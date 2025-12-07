from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a swirling vortex through a
    crumbling glass tunnel. The goal is to survive as long as possible and
    reach the end of the tunnel by avoiding collisions with falling glass
    fragments and the tunnel walls.

    The environment features high-quality visual effects, including a dynamic
    vortex, particle-based thrusters, and a procedurally generated,
    collapsing world. The physics are designed for a satisfying "game feel"
    rather than strict realism.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a swirling vortex through a crumbling glass tunnel, avoiding falling fragments and tunnel walls to survive as long as possible."
    )
    user_guide = "Use ↑↓←→ arrow keys to apply thrust. Press space for a directional burst."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TUNNEL_LENGTH = 5000  # Total length of the world in pixels
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_VORTEX_CORE = (100, 150, 255)
    COLOR_VORTEX_OUTER = (50, 80, 200)
    COLOR_THRUSTER = (255, 180, 80)
    COLOR_FRAGMENT = (180, 180, 200)
    COLOR_TUNNEL = (220, 220, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_PROGRESS_BAR = (0, 200, 255)

    # Physics
    GRAVITY = 0.15
    THRUST_POWER = 0.4
    BURST_POWER = 8.0
    DRAG = 0.985
    MAX_VELOCITY = 10

    # Player
    PLAYER_RADIUS = 15

    # World
    INITIAL_COLLAPSE_INTERVAL = 80  # Steps between collapses
    DIFFICULTY_INTERVAL = 200 # Steps to increase difficulty
    DIFFICULTY_MULTIPLIER = 0.95 # Speeds up collapse rate

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
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 50)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.last_move_dir = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        self.fragments = []
        self.particles = []
        self.tunnel_y_top = []
        self.tunnel_y_bottom = []
        self.collapse_timer = 0
        self.collapse_interval = 0
        self.prev_space_held = False
        self.win = False

        self.rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([2.0, 0.0])
        self.last_move_dir = np.array([1.0, 0.0])

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.camera_x = 0

        self.fragments = []
        self.particles = []

        self.collapse_interval = self.INITIAL_COLLAPSE_INTERVAL
        self.collapse_timer = self.collapse_interval

        self.prev_space_held = False

        self._generate_tunnel()

        return self._get_observation(), self._get_info()

    def _generate_tunnel(self):
        self.tunnel_y_top = np.zeros(self.TUNNEL_LENGTH)
        self.tunnel_y_bottom = np.zeros(self.TUNNEL_LENGTH)
        
        amplitude = 50
        frequency = 0.005
        offset = self.SCREEN_HEIGHT / 2
        min_gap = 180

        for x in range(self.TUNNEL_LENGTH):
            base_y = amplitude * math.sin(frequency * x) + offset
            self.tunnel_y_top[x] = base_y - min_gap / 2
            self.tunnel_y_bottom[x] = base_y + min_gap / 2

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0
        self._handle_input(movement, space_held)
        
        self._update_player()
        self._update_world()
        
        collision_penalty, terminated_by_collision = self._check_collisions()
        reward += collision_penalty
        
        self.steps += 1
        
        # Win condition
        if self.player_pos[0] >= self.TUNNEL_LENGTH - self.PLAYER_RADIUS:
            self.win = True
            self.game_over = True
            reward += 100.0
            
        # Max steps condition
        terminated_by_steps = self.steps >= self.MAX_STEPS
        
        terminated = self.game_over or terminated_by_steps

        # Survival reward
        if not terminated:
            reward += 0.1
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Movement thrusters
        thrust_vector = np.array([0.0, 0.0])
        if movement == 1:  # Up
            thrust_vector[1] = -1.0
        elif movement == 2:  # Down
            thrust_vector[1] = 1.0
        elif movement == 3:  # Left
            thrust_vector[0] = -1.0
        elif movement == 4:  # Right
            thrust_vector[0] = 1.0

        if np.linalg.norm(thrust_vector) > 0:
            self.player_vel += thrust_vector * self.THRUST_POWER
            self.last_move_dir = thrust_vector
            self._spawn_particles(self.player_pos.copy(), -thrust_vector, 15, self.COLOR_THRUSTER)

        # Space bar burst
        if space_held and not self.prev_space_held:
            burst_dir = self.last_move_dir if np.linalg.norm(self.last_move_dir) > 0 else np.array([1.0, 0.0])
            self.player_vel += burst_dir * self.BURST_POWER
            self._spawn_particles(self.player_pos.copy(), -burst_dir, 50, (255, 255, 150), life=60, speed_mult=2.0)
        self.prev_space_held = space_held
        
    def _update_player(self):
        # Apply gravity - REMOVED for stability test. Gravity only affects fragments.
        # self.player_vel[1] += self.GRAVITY
        # Apply drag
        self.player_vel *= self.DRAG
        # Clamp velocity
        vel_norm = np.linalg.norm(self.player_vel)
        if vel_norm > self.MAX_VELOCITY:
            self.player_vel = self.player_vel / vel_norm * self.MAX_VELOCITY
        
        # Update position
        self.player_pos += self.player_vel
        
        # Prevent going backwards
        self.player_pos[0] = max(self.player_pos[0], self.PLAYER_RADIUS)

        # Update camera
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 3

    def _update_world(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        # Update fragments
        self.fragments = [f for f in self.fragments if f['pos'][1] < self.SCREEN_HEIGHT + 50]
        for f in self.fragments:
            f['vel'][1] += 0.5  # Fragment gravity
            f['pos'] += f['vel']
            f['angle'] += f['rot_speed']

        # Handle tunnel collapse
        self.collapse_timer -= 1
        if self.collapse_timer <= 0:
            self.collapse_timer = self.collapse_interval
            # Collapse a section just off-screen
            collapse_x = self.camera_x + self.SCREEN_WIDTH + 50
            if collapse_x < self.TUNNEL_LENGTH - 100:
                self._create_fragments(collapse_x)
        
        # Increase difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.collapse_interval = max(20, self.collapse_interval * self.DIFFICULTY_MULTIPLIER)

    def _create_fragments(self, x_pos):
        tunnel_top = self.tunnel_y_top[int(x_pos)]
        tunnel_bottom = self.tunnel_y_bottom[int(x_pos)]

        for _ in range(random.randint(4, 7)):
            start_y = random.choice([tunnel_top, tunnel_bottom])
            fragment = {
                'pos': np.array([x_pos + random.uniform(-30, 30), start_y]),
                'vel': np.array([random.uniform(-1, 1), random.uniform(0, 2)]),
                'size': random.uniform(8, 15),
                'angle': random.uniform(0, 360),
                'rot_speed': random.uniform(-5, 5)
            }
            self.fragments.append(fragment)

    def _check_collisions(self):
        reward_penalty = 0.0
        terminated = False
        
        # Player vs Fragments
        for f in self.fragments:
            dist = np.linalg.norm(self.player_pos - f['pos'])
            if dist < self.PLAYER_RADIUS + f['size']:
                reward_penalty = -5.0
                terminated = True
                self._spawn_particles(self.player_pos, (f['pos'] - self.player_pos), 40, self.COLOR_FRAGMENT, life=40)
                break
        
        if terminated:
            self.game_over = True
            return reward_penalty, terminated
            
        # Player vs Tunnel Walls
        player_x_int = int(self.player_pos[0])
        if 0 <= player_x_int < self.TUNNEL_LENGTH:
            top_wall_y = self.tunnel_y_top[player_x_int]
            bottom_wall_y = self.tunnel_y_bottom[player_x_int]

            if self.player_pos[1] - self.PLAYER_RADIUS < top_wall_y or \
               self.player_pos[1] + self.PLAYER_RADIUS > bottom_wall_y:
                reward_penalty = -10.0
                terminated = True
                self._spawn_particles(self.player_pos, np.array([0,0]), 60, self.COLOR_TUNNEL, life=50)

        if terminated:
            self.game_over = True
            return reward_penalty, terminated
            
        return reward_penalty, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_tunnel()
        self._render_particles()
        self._render_fragments()
        self._render_player()

    def _render_tunnel(self):
        start_x = max(0, int(self.camera_x))
        end_x = min(self.TUNNEL_LENGTH, int(self.camera_x + self.SCREEN_WIDTH + 2))
        
        for x in range(start_x, end_x):
            screen_x = int(x - self.camera_x)
            
            # Top wall
            y_top = int(self.tunnel_y_top[x])
            pygame.draw.line(self.screen, self.COLOR_TUNNEL, (screen_x, y_top), (screen_x, y_top - 3), 2)
            
            # Bottom wall
            y_bottom = int(self.tunnel_y_bottom[x])
            pygame.draw.line(self.screen, self.COLOR_TUNNEL, (screen_x, y_bottom), (screen_x, y_bottom + 3), 2)
            
    def _render_fragments(self):
        for f in self.fragments:
            screen_pos = f['pos'] - np.array([self.camera_x, 0])
            if -50 < screen_pos[0] < self.SCREEN_WIDTH + 50:
                size = int(f['size'])
                points = []
                for i in range(5):
                    angle = math.radians(f['angle'] + i * 72)
                    points.append((
                        int(screen_pos[0] + size * math.cos(angle)),
                        int(screen_pos[1] + size * math.sin(angle))
                    ))
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_FRAGMENT)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FRAGMENT)

    def _render_player(self):
        screen_pos = self.player_pos - np.array([self.camera_x, 0])
        x, y = int(screen_pos[0]), int(screen_pos[1])
        
        # Swirling effect
        time_angle = self.steps * 0.1
        for i in range(5):
            angle = time_angle + i * (2 * math.pi / 5)
            offset_x = self.PLAYER_RADIUS * 0.5 * math.cos(angle)
            offset_y = self.PLAYER_RADIUS * 0.5 * math.sin(angle)
            
            # Outer, darker circles
            r_outer = int(self.PLAYER_RADIUS * 0.8)
            color_outer = (*self.COLOR_VORTEX_OUTER, 80)
            pygame.gfxdraw.filled_circle(self.screen, x + int(offset_x), y + int(offset_y), r_outer, color_outer)
            pygame.gfxdraw.aacircle(self.screen, x + int(offset_x), y + int(offset_y), r_outer, color_outer)

        # Core, brighter circle
        r_core = int(self.PLAYER_RADIUS * 0.9)
        color_core = (*self.COLOR_VORTEX_CORE, 150)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r_core, color_core)
        pygame.gfxdraw.aacircle(self.screen, x, y, r_core, color_core)

    def _spawn_particles(self, pos, base_vel, count, color, life=30, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            vel = base_vel + np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(int(life*0.5), life),
                'size': random.uniform(2, 5),
                'color': color
            })

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p['pos'] - np.array([self.camera_x, 0])
            if 0 < screen_pos[0] < self.SCREEN_WIDTH and 0 < screen_pos[1] < self.SCREEN_HEIGHT:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p['size']), color)

    def _render_ui(self):
        # Progress bar
        progress = self.player_pos[0] / self.TUNNEL_LENGTH
        bar_width = int(progress * self.SCREEN_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (0, 0, bar_width, 5))

        # Score text
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 15))
        
        # Steps text
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 15))

        # Game Over / Win message
        if self.game_over:
            message = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
            "win": self.win
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display screen for manual play
    # This will fail if you run headless, but is useful for local dev
    try:
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Vortex Tunnel")
    except pygame.error:
        print("Could not create display. Running headlessly.")
        display_screen = None
    
    total_reward = 0
    
    while not terminated:
        # --- Manual Control ---
        action = [0, 0, 0] # Default no-op
        if display_screen:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] else 0

            action = [movement, space_held, shift_held]
        
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to Display ---
        if display_screen:
            # The observation is (H, W, C), but pygame surface wants (W, H)
            # and surfarray.make_surface expects (W, H, C)
            obs_transposed = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(obs_transposed)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()