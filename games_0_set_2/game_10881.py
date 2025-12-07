import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:08:11.531296
# Source Brief: brief_00881.md
# Brief Index: 881
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a transforming cube through a field of moving obstacles to reach the exit. "
        "Change your form to alter your size and speed to adapt to the challenge."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to change your cube's form, "
        "which alters its size and speed."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS

    # Colors (Neon on Dark)
    COLOR_BG = (16, 16, 24)
    COLOR_WALLS = (220, 220, 255)
    COLOR_EXIT = (0, 255, 128)
    COLOR_OBSTACLE = (255, 100, 0)
    
    PLAYER_COLORS = [(0, 150, 255), (255, 255, 0), (255, 50, 50)] # Blue, Yellow, Red
    PLAYER_SIZES = [12, 18, 24]
    PLAYER_SPEEDS = [4.5, 3.5, 2.5]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_form = 0 # 0:small, 1:medium, 2:large
        
        self.obstacles = []
        self.obstacle_base_speed = 1.5
        self.obstacle_current_speed = 1.5
        
        self.exit_pos = pygame.Vector2(self.WIDTH - 40, 40)
        self.exit_size = 20
        
        self.particles = []
        
        self.transform_cooldown = 0
        self.prev_space_held = False

        self.prev_dist_to_exit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(40, self.HEIGHT - 40)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_form = 0
        
        self.obstacle_current_speed = self.obstacle_base_speed
        self._generate_obstacles(num_pairs=4)
        
        self.particles = []
        self.transform_cooldown = 0
        self.prev_space_held = False
        
        self.prev_dist_to_exit = self.player_pos.distance_to(self.exit_pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing and return the final state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_logic()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        space_held = space_held == 1

        # --- Movement ---
        move_force = pygame.Vector2(0, 0)
        if movement == 1: # Up
            move_force.y = -1
        elif movement == 2: # Down
            move_force.y = 1
        elif movement == 3: # Left
            move_force.x = -1
        elif movement == 4: # Right
            move_force.x = 1
        
        if move_force.length() > 0:
            move_force.normalize_ip()
            player_speed = self.PLAYER_SPEEDS[self.player_form]
            self.player_vel += move_force * player_speed * 0.4 # Acceleration factor

        # --- Transformation ---
        self.transform_cooldown = max(0, self.transform_cooldown - 1)
        if space_held and not self.prev_space_held and self.transform_cooldown == 0:
            # SFX: play_transform_sound()
            self.player_form = (self.player_form + 1) % 3
            self.transform_cooldown = 10 # 1/3 second cooldown
            self._create_particles(self.player_pos, self.PLAYER_COLORS[self.player_form], 20, 4)
        
        self.prev_space_held = space_held
        
    def _update_game_logic(self):
        self.steps += 1
        
        # --- Update Player ---
        self.player_vel *= 0.85 # Damping
        if self.player_vel.length() < 0.1:
            self.player_vel = pygame.Vector2(0, 0)
        self.player_pos += self.player_vel

        player_size = self.PLAYER_SIZES[self.player_form]
        self.player_pos.x = np.clip(self.player_pos.x, player_size, self.WIDTH - player_size)
        self.player_pos.y = np.clip(self.player_pos.y, player_size, self.HEIGHT - player_size)

        # --- Update Obstacles ---
        for obs in self.obstacles:
            obs['pos'] += obs['vel']
            # Bounce off walls
            if obs['pos'].x <= obs['size'] or obs['pos'].x >= self.WIDTH - obs['size']:
                obs['vel'].x *= -1
            if obs['pos'].y <= obs['size'] or obs['pos'].y >= self.HEIGHT - obs['size']:
                obs['vel'].y *= -1
        
        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.98 # Particle friction
            
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_current_speed += 0.05
            for obs in self.obstacles:
                obs['vel'].scale_to_length(self.obstacle_current_speed)

        # --- Check Collisions & Win Condition ---
        self._check_collisions()

    def _check_collisions(self):
        player_size = self.PLAYER_SIZES[self.player_form]
        player_rect = pygame.Rect(self.player_pos.x - player_size, self.player_pos.y - player_size, player_size * 2, player_size * 2)

        # Obstacle collision
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x - obs['size'], obs['pos'].y - obs['size'], obs['size'] * 2, obs['size'] * 2)
            if player_rect.colliderect(obs_rect):
                # SFX: play_explosion_sound()
                self.game_over = True
                self.win = False
                self._create_particles(self.player_pos, (255, 255, 255), 50, 6)
                return

        # Exit condition
        exit_rect = pygame.Rect(self.exit_pos.x - self.exit_size, self.exit_pos.y - self.exit_size, self.exit_size * 2, self.exit_size * 2)
        if player_rect.colliderect(exit_rect):
            # SFX: play_win_sound()
            self.game_over = True
            self.win = True
            self._create_particles(self.player_pos, self.COLOR_EXIT, 40, 5)
            return

    def _calculate_reward(self):
        if self.game_over:
            if self.win:
                return 50.0  # Victory reward
            else:
                return -100.0 # Collision or timeout penalty
        
        # --- Shaping Rewards ---
        # Survival reward
        reward = 0.01 
        
        # Distance to exit reward
        current_dist = self.player_pos.distance_to(self.exit_pos)
        reward += (self.prev_dist_to_exit - current_dist) * 0.1
        self.prev_dist_to_exit = current_dist
        
        return reward

    def _check_termination(self):
        # Termination is caused by winning or losing by collision.
        # Truncation is caused by timeout.
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Walls ---
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # --- Draw Particles ---
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], alpha)
            size = max(1, p['size'] * (p['life'] / p['max_life']))
            self._draw_circle(self.screen, p_color, p['pos'], size)

        # --- Draw Exit ---
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        glow_size = self.exit_size + pulse * 5
        self._draw_glow_rect(self.exit_pos, (glow_size, glow_size), self.COLOR_EXIT, 15)

        # --- Draw Obstacles ---
        for obs in self.obstacles:
            self._draw_glow_rect(obs['pos'], (obs['size']*2, obs['size']*2), self.COLOR_OBSTACLE, 10)
        
        # --- Draw Player ---
        if not (self.game_over and not self.win): # Don't draw player if they lost
            player_size = self.PLAYER_SIZES[self.player_form]
            player_color = self.PLAYER_COLORS[self.player_form]
            self._draw_glow_rect(self.player_pos, (player_size*2, player_size*2), player_color, 20)

    def _render_ui(self):
        # --- Timer ---
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font.render(timer_text, True, self.COLOR_WALLS)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 15, 10))
        
        # --- Score ---
        score_text = f"SCORE: {int(self.score)}"
        text_surface = self.font.render(score_text, True, self.COLOR_WALLS)
        self.screen.blit(text_surface, (15, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win
        }

    # --- Helper & Drawing Functions ---

    def _generate_obstacles(self, num_pairs):
        self.obstacles = []
        center_y = self.HEIGHT / 2
        for _ in range(num_pairs):
            pos = pygame.Vector2(
                self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8),
                self.np_random.uniform(center_y * 0.3, center_y * 0.8)
            )
            vel_angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle))
            vel.scale_to_length(self.obstacle_current_speed)
            size = self.np_random.uniform(8, 15)
            
            # Create top obstacle
            self.obstacles.append({'pos': pos.copy(), 'vel': vel.copy(), 'size': size})
            
            # Create mirrored bottom obstacle
            mirrored_pos = pygame.Vector2(pos.x, self.HEIGHT - pos.y)
            mirrored_vel = pygame.Vector2(vel.x, -vel.y)
            self.obstacles.append({'pos': mirrored_pos, 'vel': mirrored_vel, 'size': size})

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 5)
            })

    def _draw_glow_rect(self, center, size, color, glow_amount):
        w, h = size
        x, y = center.x - w / 2, center.y - h / 2
        rect = pygame.Rect(int(x), int(y), int(w), int(h))

        for i in range(glow_amount, 0, -2):
            alpha = int(100 * (1 - i / glow_amount))
            glow_color = (*color, alpha)
            s = i * 1.5
            
            surf = pygame.Surface((w + s, h + s), pygame.SRCALPHA)
            pygame.draw.rect(surf, glow_color, surf.get_rect(), border_radius=int(s/4))
            self.screen.blit(surf, (x - s/2, y - s/2))

        pygame.gfxdraw.box(self.screen, rect, (*color, 255))
        pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_WALLS)

    @staticmethod
    def _draw_circle(surface, color, center, radius):
        """Helper to draw anti-aliased circles with alpha."""
        x, y = int(center.x), int(center.y)
        r = int(radius)
        if r <= 0: return
        
        if len(color) == 4 and color[3] < 255:
            target_rect = pygame.Rect(x-r, y-r, 2*r, 2*r)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(shape_surf, color, (r, r), r)
            surface.blit(shape_surf, target_rect)
        else:
            pygame.gfxdraw.filled_circle(surface, x, y, r, color)
            pygame.gfxdraw.aacircle(surface, x, y, r, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Transforming Cube Maze")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # This action is unused in the current env logic

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Win: {info.get('win', False)}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)
        
    env.close()