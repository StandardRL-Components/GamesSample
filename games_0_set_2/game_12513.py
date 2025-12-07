import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player launches and juggles three balls.

    The goal is to keep all three balls airborne for 60 seconds.
    The player controls a launcher to set the angle and speed of each launch.
    Balls have different base gravities and can collect orbs to increase their mass,
    making them heavier and harder to keep in the air.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=none, 1=up (power+), 2=down (power-), 3=left (angle-), 4=right (angle+)
    - `actions[1]` (Space): 0=released, 1=held (launches a ball on press)
    - `actions[2]` (Shift): 0=released, 1=held (no effect)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.01 per airborne ball per step.
    - +5 for collecting an orb.
    - +100 for winning (surviving 60 seconds).
    - -100 for losing (a ball hits the floor).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch and juggle three balls with varying properties. Collect orbs to increase their mass, but be careful, "
        "as heavier balls are harder to keep airborne. Survive for 60 seconds to win!"
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim the launcher and ↑↓ keys to adjust power. Press space to launch a ball."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = FPS * 60  # 60 seconds

    # Colors
    COLOR_BG = (15, 20, 45)
    COLOR_LAUNCHER = (220, 220, 240)
    COLOR_ORB = (80, 255, 150)
    COLOR_PARTICLE = (255, 200, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)

    BALL_COLORS = [
        (100, 150, 255),  # Mass 1 (Blue)
        (255, 220, 100),  # Mass 2 (Yellow)
        (255, 100, 100),  # Mass 3 (Red)
        (255, 150, 255),  # Mass 4+ (Magenta)
    ]

    # Physics & Gameplay
    GRAVITY = 0.1
    LAUNCHER_POS = (WIDTH // 2, HEIGHT - 20)
    MIN_LAUNCH_ANGLE, MAX_LAUNCH_ANGLE = 190, 350  # In degrees, 180 is left, 270 is up
    MIN_LAUNCH_POWER, MAX_LAUNCH_POWER = 4.0, 14.0
    ANGLE_STEP = 1.5
    POWER_STEP = 0.2
    WALL_ELASTICITY = 0.9
    MAX_ORBS = 5
    ORB_RADIUS = 8
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_timer = pygame.font.SysFont('Consolas', 30, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launcher_angle = 0
        self.launcher_power = 0
        self.balls_to_launch = []
        self.active_balls = []
        self.orbs = []
        self.particles = []
        self.prev_space_held = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Launcher state
        self.launcher_angle = (self.MIN_LAUNCH_ANGLE + self.MAX_LAUNCH_ANGLE) / 2
        self.launcher_power = (self.MIN_LAUNCH_POWER + self.MAX_LAUNCH_POWER) / 2
        self.prev_space_held = False

        # Define the three balls to be launched
        self.balls_to_launch = [
            {'base_gravity': 1.0, 'name': 'Light'},
            {'base_gravity': 1.2, 'name': 'Medium'},
            {'base_gravity': 1.4, 'name': 'Heavy'},
        ]
        
        self.active_balls = []
        self.particles = []
        
        # Initial orb spawning
        self.orbs = []
        for _ in range(self.MAX_ORBS):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- 1. Handle Input & Update Launcher ---
        self._handle_input(action)

        # --- 2. Update Game Logic & Physics ---
        self._update_physics()
        
        # --- 3. Handle Collisions ---
        orb_reward = self._handle_collisions()
        reward += orb_reward

        # --- 4. Spawn New Entities ---
        self._update_spawners()

        # --- 5. Calculate Rewards ---
        # Airtime reward
        reward += 0.01 * len(self.active_balls)
        self.score += orb_reward # Only score points for orbs

        # --- 6. Check Termination Conditions ---
        self.steps += 1
        is_win = self.steps >= self.MAX_STEPS and not self.game_over
        is_loss = any(b['pos'].y > self.HEIGHT + b['radius'] for b in self.active_balls)

        if is_loss and not self.game_over:
            terminated = True
            reward -= 100
            self.game_over = True
        elif is_win and not self.game_over:
            terminated = True
            reward += 100
            self.score += 1000 # Victory bonus
            self.game_over = True

        truncated = False
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_over:
            return

        movement, space_held, _ = action
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # Update launcher only if there are balls left to launch
        if self.balls_to_launch:
            if movement == 1: # Up
                self.launcher_power = min(self.MAX_LAUNCH_POWER, self.launcher_power + self.POWER_STEP)
            elif movement == 2: # Down
                self.launcher_power = max(self.MIN_LAUNCH_POWER, self.launcher_power - self.POWER_STEP)
            elif movement == 3: # Left
                self.launcher_angle = max(self.MIN_LAUNCH_ANGLE, self.launcher_angle - self.ANGLE_STEP)
            elif movement == 4: # Right
                self.launcher_angle = min(self.MAX_LAUNCH_ANGLE, self.launcher_angle + self.ANGLE_STEP)

            if space_pressed:
                self._launch_ball()

    def _launch_ball(self):
        if not self.balls_to_launch:
            return
            
        ball_spec = self.balls_to_launch.pop(0)
        angle_rad = math.radians(self.launcher_angle)
        
        velocity = pygame.math.Vector2(
            math.cos(angle_rad) * self.launcher_power,
            math.sin(angle_rad) * self.launcher_power
        )
        
        new_ball = {
            'pos': pygame.math.Vector2(self.LAUNCHER_POS),
            'vel': velocity,
            'mass': 1,
            'radius': 12,
            'base_gravity': ball_spec['base_gravity'],
            'name': ball_spec['name'],
            'trail': []
        }
        self.active_balls.append(new_ball)

    def _update_physics(self):
        # Update balls
        for ball in self.active_balls:
            effective_gravity = self.GRAVITY * ball['base_gravity'] * (1 + (ball['mass']-1) * 0.5)
            ball['vel'].y += effective_gravity
            ball['pos'] += ball['vel']

            # Wall bounces
            if ball['pos'].x < ball['radius'] or ball['pos'].x > self.WIDTH - ball['radius']:
                ball['vel'].x *= -self.WALL_ELASTICITY
                ball['pos'].x = np.clip(ball['pos'].x, ball['radius'], self.WIDTH - ball['radius'])
            if ball['pos'].y < ball['radius']:
                ball['vel'].y *= -self.WALL_ELASTICITY
                ball['pos'].y = np.clip(ball['pos'].y, ball['radius'], self.HEIGHT)
            
            # Update trail
            ball['trail'].append(tuple(ball['pos']))
            if len(ball['trail']) > 15:
                ball['trail'].pop(0)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        for ball in self.active_balls:
            for orb in self.orbs[:]:
                dist = ball['pos'].distance_to(orb['pos'])
                if dist < ball['radius'] + self.ORB_RADIUS:
                    self.orbs.remove(orb)
                    ball['mass'] += 1
                    ball['radius'] = 12 + (ball['mass'] - 1) * 2
                    reward += 5
                    self.score += 50
                    self._create_particles(orb['pos'])
                    return reward # Return early to avoid multiple collections in one frame
        return reward

    def _spawn_orb(self):
        pos = pygame.math.Vector2(
            self.np_random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS),
            self.np_random.uniform(self.ORB_RADIUS, self.HEIGHT * 0.75) # Spawn in upper 3/4
        )
        self.orbs.append({'pos': pos})

    def _update_spawners(self):
        if len(self.orbs) < self.MAX_ORBS and self.np_random.random() < 0.02:
            self._spawn_orb()

    def _create_particles(self, pos, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 31),
                'radius': self.np_random.uniform(1, 4),
                'color': self.COLOR_PARTICLE
            })

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
            "active_balls": len(self.active_balls),
            "balls_left": len(self.balls_to_launch),
        }

    def _draw_glow_circle(self, surface, color, pos, radius, glow_strength=1.5, glow_alpha=70):
        """Draws a circle with a glowing effect."""
        int_pos = (int(pos.x), int(pos.y))
        
        # Draw the glow
        glow_radius = int(radius * glow_strength)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int_pos[0] - glow_radius, int_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw the main circle (anti-aliased)
        pygame.gfxdraw.aacircle(surface, int_pos[0], int_pos[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int_pos[0], int_pos[1], int(radius), color)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), max(0, int(p['radius'] * (p['life']/30.0))))

        # Draw orbs
        for orb in self.orbs:
            self._draw_glow_circle(self.screen, self.COLOR_ORB, orb['pos'], self.ORB_RADIUS)

        # Draw balls
        for ball in self.active_balls:
            # Trail
            if len(ball['trail']) > 1:
                trail_color_idx = min(len(self.BALL_COLORS) - 1, ball['mass'] - 1)
                trail_color = self.BALL_COLORS[trail_color_idx]
                for i, p in enumerate(ball['trail']):
                    alpha = int(255 * (i / len(ball['trail'])))
                    pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), int(ball['radius'] * (i / len(ball['trail']))), (*trail_color, alpha))

            # Ball itself
            color_idx = min(len(self.BALL_COLORS) - 1, ball['mass'] - 1)
            color = self.BALL_COLORS[color_idx]
            self._draw_glow_circle(self.screen, color, ball['pos'], ball['radius'], glow_alpha=90)

        # Draw launcher
        if self.balls_to_launch:
            # Aiming line
            angle_rad = math.radians(self.launcher_angle)
            power_ratio = (self.launcher_power - self.MIN_LAUNCH_POWER) / (self.MAX_LAUNCH_POWER - self.MIN_LAUNCH_POWER)
            line_len = 30 + 40 * power_ratio
            end_pos = (
                self.LAUNCHER_POS[0] + math.cos(angle_rad) * line_len,
                self.LAUNCHER_POS[1] + math.sin(angle_rad) * line_len
            )
            pygame.draw.line(self.screen, self.COLOR_LAUNCHER, self.LAUNCHER_POS, end_pos, 3)
            pygame.draw.circle(self.screen, self.COLOR_LAUNCHER, (int(end_pos[0]), int(end_pos[1])), 5)

            # Base
            pygame.draw.circle(self.screen, self.COLOR_LAUNCHER, self.LAUNCHER_POS, 8)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_str = f"{minutes:02}:{seconds:02}"
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Next ball display
        if self.balls_to_launch:
            next_ball_spec = self.balls_to_launch[0]
            next_ball_text = self.font_ui.render(f"NEXT: {next_ball_spec['name']} (G: {next_ball_spec['base_gravity']:.1f}x)", True, self.COLOR_UI_TEXT)
            self.screen.blit(next_ball_text, (self.WIDTH // 2 - next_ball_text.get_width() // 2, self.HEIGHT - 50))
        elif not self.active_balls and not self.game_over:
             # This case should not happen if logic is correct, but as a fallback
             pass
        
        # Game over message
        if self.game_over:
            is_win = self.steps >= self.MAX_STEPS
            msg = "VICTORY!" if is_win else "GAME OVER"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            
            end_text = self.font_timer.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = end_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, end_rect)


    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # The environment must be runnable headless
    # but for testing, we can use a local display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Ball Juggler")
    clock = pygame.time.Clock()

    obs, info = env.reset(seed=42)
    done = False
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            done = True
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause to see end screen
            obs, info = env.reset(seed=42)

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()