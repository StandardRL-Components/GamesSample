import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:27:06.518651
# Source Brief: brief_02413.md
# Brief Index: 2413
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bouncing ball to dodge descending spikes.

    The goal is to navigate the ball to the bottom of the screen.
    Each collision with a spike shrinks the ball, making it harder to control but
    also providing a small upward bounce. The episode ends if the ball's radius
    reaches zero, if it reaches the goal, or after a maximum number of steps.

    The visual style is minimalist and geometric, prioritizing clarity and game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a bouncing ball to dodge ascending spikes and reach the goal at the bottom. "
        "Each collision shrinks the ball, making it harder to control."
    )
    user_guide = "Use the ← and → arrow keys to move the ball left and right."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_GLOW = (255, 120, 120)
    COLOR_SPIKE = (220, 220, 255)
    COLOR_GOAL = (180, 255, 180)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (200, 200, 220)

    # Physics and Gameplay
    GRAVITY = 0.4
    HORIZONTAL_ACCEL = 0.5
    HORIZONTAL_DRAG = 0.98
    INITIAL_BALL_RADIUS = 25
    MIN_BALL_RADIUS = 2
    SPIKE_HIT_RADIUS_LOSS = 2.5
    SPIKE_HIT_BOUNCE = -4.0
    INITIAL_SPIKE_SPEED = 1.5
    SPIKE_SPEED_INCREASE_INTERVAL = 200
    SPIKE_SPEED_INCREASE_AMOUNT = 0.1
    SPIKE_HEIGHT = 20
    SPIKE_GAP_WIDTH = 140
    SPIKE_SPAWN_Y_INTERVAL = 120
    MAX_STEPS = 2000
    GOAL_Y = SCREEN_HEIGHT - 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_radius = 0.0
        
        self.spikes = []
        self.spike_speed = 0.0
        self.last_spike_spawn_y = 0.0
        
        self.particles = []

        # self.reset() is called by the wrapper or user, not needed here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Ball state
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2.0, self.INITIAL_BALL_RADIUS * 2.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_radius = float(self.INITIAL_BALL_RADIUS)
        
        # Obstacle state
        self.spikes = []
        self.spike_speed = self.INITIAL_SPIKE_SPEED
        self.last_spike_spawn_y = 0.0
        self._spawn_initial_spikes()
        
        # Effects state
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_ball()
        self._update_spikes()
        reward += self._handle_collisions()
        self._update_particles()
        self._update_difficulty()
        
        self.score += reward
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.ball_radius < self.MIN_BALL_RADIUS:
            terminated = True
            # No extra penalty, the collision penalty is sufficient
        elif self.ball_pos[1] + self.ball_radius >= self.GOAL_Y:
            terminated = True
            reward += 100.0  # Goal reward
            self.score += 100.0
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3:  # Left
            self.ball_vel[0] -= self.HORIZONTAL_ACCEL
        elif movement == 4: # Right
            self.ball_vel[0] += self.HORIZONTAL_ACCEL
            
    def _update_ball(self):
        # Apply physics
        self.ball_vel[1] += self.GRAVITY
        self.ball_vel[0] *= self.HORIZONTAL_DRAG
        self.ball_pos += self.ball_vel

        # Wall bounces
        if self.ball_pos[0] - self.ball_radius < 0:
            self.ball_pos[0] = self.ball_radius
            self.ball_vel[0] *= -0.8
        elif self.ball_pos[0] + self.ball_radius > self.SCREEN_WIDTH:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.ball_radius
            self.ball_vel[0] *= -0.8
            
        if self.ball_pos[1] - self.ball_radius < 0:
            self.ball_pos[1] = self.ball_radius
            self.ball_vel[1] *= -0.5 # Dampened bounce on ceiling

    def _update_spikes(self):
        # Move existing spikes
        for spike in self.spikes:
            spike.y += self.spike_speed
        
        # Remove off-screen spikes
        self.spikes = [s for s in self.spikes if s.top < self.SCREEN_HEIGHT]
        
        # Spawn new spikes
        if not self.spikes or self.spikes[-1].y > self.SPIKE_SPAWN_Y_INTERVAL:
            self._spawn_spike_row()

    def _spawn_initial_spikes(self):
        for y_pos in range(int(self.SCREEN_HEIGHT * 0.4), self.SCREEN_HEIGHT, self.SPIKE_SPAWN_Y_INTERVAL):
             self._spawn_spike_row(y_pos)

    def _spawn_spike_row(self, y_pos=None):
        if y_pos is None:
            y_pos = -self.SPIKE_HEIGHT

        gap_x = self.np_random.uniform(self.ball_radius, self.SCREEN_WIDTH - self.SPIKE_GAP_WIDTH - self.ball_radius)
        
        spike1 = pygame.Rect(0, y_pos, gap_x, self.SPIKE_HEIGHT)
        spike2 = pygame.Rect(gap_x + self.SPIKE_GAP_WIDTH, y_pos, 
                             self.SCREEN_WIDTH - (gap_x + self.SPIKE_GAP_WIDTH), self.SPIKE_HEIGHT)
        
        self.spikes.extend([spike1, spike2])
        
    def _handle_collisions(self):
        collision_reward = 0.0
        collided = False
        
        spikes_to_remove = []

        for i, spike in enumerate(self.spikes):
            # Find the closest point on the rect to the circle's center
            closest_x = max(spike.left, min(self.ball_pos[0], spike.right))
            closest_y = max(spike.top, min(self.ball_pos[1], spike.bottom))
            
            distance_sq = (self.ball_pos[0] - closest_x)**2 + (self.ball_pos[1] - closest_y)**2
            
            if distance_sq < self.ball_radius**2:
                collided = True
                # Assuming spikes are always in pairs, mark both for removal
                pair_idx = i + 1 if i % 2 == 0 else i - 1
                if i not in spikes_to_remove:
                    spikes_to_remove.append(i)
                if pair_idx not in spikes_to_remove and 0 <= pair_idx < len(self.spikes):
                     spikes_to_remove.append(pair_idx)
        
        if collided:
            # Remove spikes in reverse index order to avoid shifting issues
            for index in sorted(spikes_to_remove, reverse=True):
                del self.spikes[index]
            
            collision_reward = -5.0
            self.ball_radius = max(self.MIN_BALL_RADIUS, self.ball_radius - self.SPIKE_HIT_RADIUS_LOSS)
            self.ball_vel[1] = self.SPIKE_HIT_BOUNCE
            self._create_particles(self.ball_pos, 20)
        
        return collision_reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': 1.0,
                'size': self.np_random.uniform(1, 3)
            })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPIKE_SPEED_INCREASE_INTERVAL == 0:
            self.spike_speed += self.SPIKE_SPEED_INCREASE_AMOUNT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ball_radius": self.ball_radius,
            "spike_speed": self.spike_speed,
        }

    def _render_game(self):
        # Render Goal Line
        pygame.draw.line(self.screen, self.COLOR_GOAL, (0, self.GOAL_Y), (self.SCREEN_WIDTH, self.GOAL_Y), 3)

        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 255)))
            color = (*self.COLOR_PARTICLE, alpha)
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render Spikes
        for spike in self.spikes:
            # Draw spike as a filled rectangle
            pygame.draw.rect(self.screen, self.COLOR_SPIKE, spike)
            
            # Draw triangle pattern on spikes for better visual
            if spike.width > 10:
                num_triangles = int(spike.width / self.SPIKE_HEIGHT)
                for i in range(num_triangles):
                    base_x = spike.left + i * self.SPIKE_HEIGHT
                    p1 = (base_x, spike.bottom)
                    p2 = (base_x + self.SPIKE_HEIGHT, spike.bottom)
                    p3 = (base_x + self.SPIKE_HEIGHT / 2, spike.top)
                    pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_BG)


        # Render Ball
        if self.ball_radius >= self.MIN_BALL_RADIUS:
            x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
            r = int(self.ball_radius)
            
            # Glow effect
            glow_radius = int(r * 1.2)
            glow_alpha = 80
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_BALL_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))

            # Main ball with anti-aliasing
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        # Health bar (visualized as ball size)
        health_ratio = max(0, (self.ball_radius - self.MIN_BALL_RADIUS) / (self.INITIAL_BALL_RADIUS - self.MIN_BALL_RADIUS))
        bar_width = 150
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 15
        
        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_width, bar_height))
        fill_width = bar_width * health_ratio
        fill_color = self.COLOR_BALL if health_ratio > 0.25 else (255, 50, 50)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_height))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # To play, you might need to comment out the os.environ line at the top
    # or run in an environment where a display is available.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Check if a display is available before creating one
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Bounce Dodge")
        has_display = True
    except pygame.error:
        print("No display available. Running headlessly.")
        has_display = False

    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print("Left/Right Arrows: Move Ball")
    print("Q: Quit")

    running = True
    while running:
        action = np.array([0, 0, 0])  # Default action: no-op
        
        if has_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
        else: # Simple auto-play for headless mode
            if env.steps > 1000:
                running = False
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if has_display:
            # Draw the observation from the environment to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            if not has_display: # End after one episode in headless mode
                running = False
            
        clock.tick(60) # Run at 60 FPS for smooth human play
        
    env.close()