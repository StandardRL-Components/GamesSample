import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:28:46.073545
# Source Brief: brief_01756.md
# Brief Index: 1756
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a chain and navigate it to collect targets, growing longer with each one. "
        "Avoid obstacles and reach the target length before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ or ↑↓ arrow keys to aim. Press space to launch. "
        "Hold shift while launching for a speed boost."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_TIME = 45  # seconds
    WIN_LENGTH = 30

    # Colors (Neon on Dark)
    COLOR_BG = (10, 15, 25)
    COLOR_CHAIN = (50, 255, 100)
    COLOR_CHAIN_GLOW = (50, 255, 100, 50)
    COLOR_TARGET = (100, 200, 255)
    COLOR_TARGET_GLOW = (100, 200, 255, 60)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 60)
    COLOR_PARTICLE = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_AIM = (255, 255, 255, 100)

    # Physics
    BASE_LAUNCH_SPEED = 7.0
    SHIFT_SPEED_BONUS = 1.5
    DRAG = 0.99
    SEGMENT_LENGTH = 8.0
    ANGLE_ADJUST_RATE = 0.05  # radians per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.chain_segments = None
        self.head_velocity = None
        self.is_launched = None
        self.aim_angle = None
        self.targets = None
        self.obstacles = None
        self.particles = None
        self.last_space_state = None
        self.last_dist_to_closest_target = None
        self.last_dist_to_closest_obstacle = None
        self.win_condition_met = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.timer = self.MAX_TIME
        self.max_steps = self.MAX_TIME * self.FPS

        initial_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 30)
        self.chain_segments = [initial_pos.copy() for _ in range(5)]
        self.head_velocity = pygame.Vector2(0, 0)

        self.is_launched = False
        self.aim_angle = -math.pi / 2  # Straight up

        self.targets = []
        self.obstacles = []
        self._spawn_initial_entities()

        self.particles = []
        self.last_space_state = 0

        if self.targets:
            self.last_dist_to_closest_target = min(
                (self.chain_segments[0] - pygame.Vector2(t.center)).length() for t in self.targets
            )
        else:
            self.last_dist_to_closest_target = float('inf')

        if self.obstacles:
            self.last_dist_to_closest_obstacle = min(
                (self.chain_segments[0] - pygame.Vector2(o.center)).length() for o in self.obstacles
            )
        else:
            self.last_dist_to_closest_obstacle = float('inf')

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()

        # Continuous distance-based rewards
        if self.is_launched:
            dist_rew = self._calculate_distance_rewards()
            reward += dist_rew
        
        event_rew = self._handle_collisions()
        reward += event_rew

        self.score += reward
        self.steps += 1
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += term_reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        if not self.is_launched:
            if movement in [1, 4]:  # Up, Right -> Rotate CCW
                self.aim_angle -= self.ANGLE_ADJUST_RATE
            elif movement in [2, 3]:  # Down, Left -> Rotate CW
                self.aim_angle += self.ANGLE_ADJUST_RATE
            self.aim_angle = self.aim_angle % (2 * math.pi)

            if space_held and not self.last_space_state:
                self.is_launched = True
                launch_speed = self.BASE_LAUNCH_SPEED
                if shift_held:
                    launch_speed *= self.SHIFT_SPEED_BONUS
                self.head_velocity = pygame.Vector2(
                    math.cos(self.aim_angle) * launch_speed,
                    math.sin(self.aim_angle) * launch_speed,
                )
                # SFX: Chain launch swoosh

        self.last_space_state = space_held

    def _update_game_state(self):
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        self._update_particles()

        if self.is_launched:
            self._update_chain()

    def _update_chain(self):
        # Update head
        self.head_velocity *= self.DRAG
        self.chain_segments[0] += self.head_velocity

        # Wall bounces for head
        if not (0 < self.chain_segments[0].x < self.WIDTH):
            self.chain_segments[0].x = np.clip(self.chain_segments[0].x, 0, self.WIDTH)
            self.head_velocity.x *= -1
        if not (0 < self.chain_segments[0].y < self.HEIGHT):
            self.chain_segments[0].y = np.clip(self.chain_segments[0].y, 0, self.HEIGHT)
            self.head_velocity.y *= -1

        # Update body segments (inverse kinematics)
        for i in range(1, len(self.chain_segments)):
            leader = self.chain_segments[i - 1]
            follower = self.chain_segments[i]
            direction = follower - leader
            dist = direction.length()

            if dist > self.SEGMENT_LENGTH:
                self.chain_segments[i] = leader + direction.normalize() * self.SEGMENT_LENGTH

    def _calculate_distance_rewards(self):
        reward = 0
        head_pos = self.chain_segments[0]

        if self.targets:
            closest_target_dist = min((head_pos - pygame.Vector2(t.center)).length() for t in self.targets)
            if closest_target_dist < self.last_dist_to_closest_target:
                reward += 0.1
            self.last_dist_to_closest_target = closest_target_dist

        if self.obstacles:
            closest_obstacle_dist = min((head_pos - pygame.Vector2(o.center)).length() for o in self.obstacles)
            if closest_obstacle_dist < self.last_dist_to_closest_obstacle:
                reward -= 0.2
            self.last_dist_to_closest_obstacle = closest_obstacle_dist
            
        return reward

    def _handle_collisions(self):
        reward = 0
        head_pos = self.chain_segments[0]
        head_radius = 5

        # Targets
        for target in self.targets[:]:
            if (head_pos - pygame.Vector2(target.center)).length() < head_radius + target.width / 2:
                reward += 1.0
                self.targets.remove(target)
                self._spawn_target()
                
                # Add new segment
                self.chain_segments.append(self.chain_segments[-1].copy())
                
                self._create_particles(target.center, 20, self.COLOR_PARTICLE)
                # SFX: Positive chime, collect sound
                break

        # Obstacles
        for obstacle in self.obstacles:
            if obstacle.collidepoint(head_pos):
                reward -= 2.0
                
                # Shrink chain
                num_to_remove = min(2, len(self.chain_segments) -1)
                if num_to_remove > 0:
                    self.chain_segments = self.chain_segments[:-num_to_remove]
                
                # Bounce
                self.head_velocity *= -0.8
                self.chain_segments[0] += self.head_velocity * 2 # Push out of obstacle
                
                self._create_particles(head_pos, 10, self.COLOR_OBSTACLE)
                # SFX: Negative buzz, impact sound
                break
        
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        
        if len(self.chain_segments) >= self.WIN_LENGTH:
            terminated = True
            terminal_reward = 100.0
            self.win_condition_met = True
            # SFX: Victory fanfare
        elif len(self.chain_segments) <= 1: # Game over if chain is too short
            terminated = True
            terminal_reward = -100.0
            # SFX: Failure sound
        elif self.timer <= 0:
            terminated = True
            terminal_reward = -100.0
            # SFX: Timeout buzzer
        
        return terminated, terminal_reward

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
            "chain_length": len(self.chain_segments),
            "timer": self.timer,
        }

    def _render_game(self):
        # Obstacles
        for obs in self.obstacles:
            self._draw_glowing_rect(self.screen, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW, obs, 10)
        
        # Targets
        for target in self.targets:
            self._draw_glowing_circle(self.screen, self.COLOR_TARGET, self.COLOR_TARGET_GLOW, target.center, target.width // 2, 15)

        # Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
            if p['lifetime'] > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])

        # Chain
        if len(self.chain_segments) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_CHAIN, False, self.chain_segments, 2)
        
        # Chain Head
        if self.chain_segments:
            head_pos = self.chain_segments[0]
            self._draw_glowing_circle(self.screen, self.COLOR_CHAIN, self.COLOR_CHAIN_GLOW, head_pos, 5, 10)

        # Aiming indicator
        if not self.is_launched:
            start_pos = self.chain_segments[0]
            end_pos = start_pos + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 50
            self._draw_dashed_line(self.screen, self.COLOR_AIM, start_pos, end_pos)

    def _render_ui(self):
        # Chain Length
        length_text = self.font_ui.render(f"LENGTH: {len(self.chain_segments):02}/{self.WIN_LENGTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(length_text, (10, 10))

        # Timer
        timer_str = f"{int(self.timer // 60):02}:{int(self.timer % 60):02}"
        timer_text = self.font_ui.render(f"TIME: {timer_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Text
        if self.game_over:
            if self.win_condition_met:
                msg = "VICTORY!"
                color = self.COLOR_CHAIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _spawn_initial_entities(self):
        occupied_areas = []
        # Add player start area
        player_start_rect = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 80, 100, 100)
        occupied_areas.append(player_start_rect)

        for _ in range(2): # Obstacles
            self._spawn_obstacle(occupied_areas)
        
        for _ in range(3): # Targets
            self._spawn_target(occupied_areas)

    def _get_valid_spawn_rect(self, size, occupied_areas):
        for _ in range(100): # Max 100 attempts
            rect = pygame.Rect(
                self.np_random.integers(30, self.WIDTH - 30 - size[0]),
                self.np_random.integers(30, self.HEIGHT - 80 - size[1]),
                size[0], size[1]
            )
            if not any(rect.colliderect(area) for area in occupied_areas):
                occupied_areas.append(rect)
                return rect
        return None # Failed to find a spot

    def _spawn_target(self, occupied_areas=None):
        if occupied_areas is None: # For respawning during gameplay
            occupied_areas = self.targets + self.obstacles
        
        size = self.np_random.integers(15, 25)
        rect = self._get_valid_spawn_rect((size, size), occupied_areas)
        if rect:
            self.targets.append(rect)

    def _spawn_obstacle(self, occupied_areas):
        size = (self.np_random.integers(40, 80), self.np_random.integers(15, 30))
        rect = self._get_valid_spawn_rect(size, occupied_areas)
        if rect:
            self.obstacles.append(rect)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color,
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    @staticmethod
    def _draw_glowing_circle(surface, color, glow_color, pos, radius, glow_radius):
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius + glow_radius), glow_color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius + glow_radius), glow_color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)

    @staticmethod
    def _draw_glowing_rect(surface, color, glow_color, rect, glow_size):
        glow_rect = rect.inflate(glow_size, glow_size)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
        surface.blit(s, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=4)
        
    @staticmethod
    def _draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=5):
        origin = pygame.Vector2(start_pos)
        target = pygame.Vector2(end_pos)
        displacement = target - origin
        length = displacement.length()
        if length == 0: return
        
        direction = displacement.normalize()
        
        current_pos = origin.copy()
        for i in range(int(length / (dash_length * 2))):
            start = current_pos
            end = current_pos + direction * dash_length
            pygame.draw.aaline(surf, color, start, end, width)
            current_pos += direction * dash_length * 2

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # For manual control, we need a Pygame window
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.set_caption("Chain Momentum")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard keys to actions for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        env.clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {total_reward:.2f}, Final Length: {info['chain_length']}")
    
    # Keep the final screen visible for a moment
    if info.get("chain_length", 0) > 0:
        pygame.time.wait(3000)

    env.close()