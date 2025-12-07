import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:06:57.964592
# Source Brief: brief_03005.md
# Brief Index: 3005
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
        "Collect all the colored pixels to win. Each pixel you collect changes your color, "
        "increases your speed, and rotates your direction 90 degrees."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to change your direction of movement."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    NUM_TARGETS = 15

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER_INITIAL = (255, 255, 255)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_GRID_BG = (40, 40, 60)
    
    # --- Game Parameters ---
    PLAYER_SIZE = 3
    PLAYER_OUTLINE_THICKNESS = 2
    TARGET_SIZE = 3
    INITIAL_PLAYER_SPEED = 120 # pixels per second
    SPEED_INCREASE_FACTOR = 1.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_speed = None
        self.player_color = None
        self.player_rect = None
        self.targets = []
        self.collected_colors = []
        self.target_colors = []
        self.effects = []
        
        # --- Initialize state ---
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # --- Reset Player ---
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        initial_direction = self.np_random.uniform(0, 360)
        self.player_vel = pygame.Vector2(1, 0).rotate(initial_direction)
        self.player_speed = self.INITIAL_PLAYER_SPEED / self.FPS
        self.player_vel.scale_to_length(self.player_speed)
        self.player_color = self.COLOR_PLAYER_INITIAL
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- Reset Targets and Colors ---
        self.collected_colors = []
        self.target_colors = self._generate_target_colors()
        self.targets = self._spawn_targets()

        # --- Reset Effects ---
        self.effects = []

        # --- Return initial observation and info ---
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0

        # --- 1. Handle Input ---
        self._handle_input(action)

        # --- 2. Update Player ---
        hit_edge = self._update_player()
        if not hit_edge:
            reward += 0.01 # Small reward for staying off the walls
        
        # --- 3. Handle Collisions ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- 4. Update Visual Effects ---
        self._update_effects()

        # --- 5. Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated:
            # Goal-oriented reward for winning
            reward += 100
        
        self.score += reward

        # --- Return 5-tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement != 0:
            # SFX: Direction Change
            if movement == 1: # Up
                self.player_vel.x, self.player_vel.y = 0, -1
            elif movement == 2: # Down
                self.player_vel.x, self.player_vel.y = 0, 1
            elif movement == 3: # Left
                self.player_vel.x, self.player_vel.y = -1, 0
            elif movement == 4: # Right
                self.player_vel.x, self.player_vel.y = 1, 0
            self.player_vel.scale_to_length(self.player_speed)

    def _update_player(self):
        self.player_pos += self.player_vel
        hit_edge = False

        # Boundary checks and bounce
        if self.player_pos.x <= 0 or self.player_pos.x >= self.WIDTH - self.PLAYER_SIZE:
            self.player_vel.x *= -1
            self.player_pos.x = max(0, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE))
            hit_edge = True
        if self.player_pos.y <= 0 or self.player_pos.y >= self.HEIGHT - self.PLAYER_SIZE:
            self.player_vel.y *= -1
            self.player_pos.y = max(0, min(self.player_pos.y, self.HEIGHT - self.PLAYER_SIZE))
            hit_edge = True
        
        if hit_edge:
            # SFX: Wall Bounce
            pass

        self.player_rect.center = self.player_pos
        return hit_edge

    def _handle_collisions(self):
        reward = 0
        collided_target = None
        
        for target in self.targets:
            if self.player_rect.colliderect(target['rect']):
                collided_target = target
                break
        
        if collided_target:
            # SFX: Pixel Collect
            self.targets.remove(collided_target)
            
            # Check if it's a new color
            if collided_target['color'] not in self.collected_colors:
                self.collected_colors.append(collided_target['color'])
                reward += 10.0 # Reward for new color
            
            # Update player state
            self.player_color = collided_target['color']
            self.player_speed *= self.SPEED_INCREASE_FACTOR
            self.player_vel.scale_to_length(self.player_speed)
            self.player_vel.rotate_ip(90) # Clockwise 90 degrees
            
            # Add visual effect
            self._add_glow_effect(self.player_pos, collided_target['color'])

        return reward

    def _update_effects(self):
        self.effects = [effect for effect in self.effects if effect.update()]

    def _check_termination(self):
        if len(self.collected_colors) >= self.NUM_TARGETS:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True # This is handled as truncation in step()
        return False

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements ---
        self._render_game()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render effects (drawn first, to be underneath other elements)
        for effect in self.effects:
            effect.draw(self.screen)

        # Render targets
        for target in self.targets:
            pygame.draw.rect(self.screen, target['color'], target['rect'])
        
        # Render player outline
        outline_size = self.PLAYER_SIZE + 2 * self.PLAYER_OUTLINE_THICKNESS
        outline_rect = pygame.Rect(0, 0, outline_size, outline_size)
        outline_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, outline_rect)

        # Render player
        self.player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.player_color, self.player_rect)

    def _render_ui(self):
        # Render collected colors grid
        grid_rows, grid_cols = 3, 5
        cell_size = 12
        padding = 3
        start_x, start_y = 10, 10
        
        for i in range(self.NUM_TARGETS):
            row = i // grid_cols
            col = i % grid_cols
            
            x = start_x + col * (cell_size + padding)
            y = start_y + row * (cell_size + padding)
            
            rect = pygame.Rect(x, y, cell_size, cell_size)
            
            if i < len(self.collected_colors):
                color = self.collected_colors[i]
            else:
                color = self.COLOR_UI_GRID_BG
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, rect, 1) # Border

        # Render text info
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 30))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "colors_collected": len(self.collected_colors),
            "player_speed": self.player_speed * self.FPS, # Back to pixels/sec
        }

    # --- Helper Methods ---
    def _generate_target_colors(self):
        colors = []
        for i in range(self.NUM_TARGETS):
            hue = int((i / self.NUM_TARGETS) * 360)
            color = pygame.Color(0)
            color.hsva = (hue, 90, 95, 100) # Bright, saturated colors
            colors.append((color.r, color.g, color.b))
        return colors

    def _spawn_targets(self):
        targets = []
        existing_rects = [pygame.Rect(self.player_pos.x - 25, self.player_pos.y - 25, 50, 50)] # Avoid spawning near player start
        
        for color in self.target_colors:
            while True:
                x = self.np_random.integers(10, self.WIDTH - 10)
                y = self.np_random.integers(10, self.HEIGHT - 10)
                new_rect = pygame.Rect(x, y, self.TARGET_SIZE, self.TARGET_SIZE)
                
                # Check for overlap with existing targets
                if not any(new_rect.colliderect(r.inflate(20, 20)) for r in existing_rects):
                    targets.append({'rect': new_rect, 'color': color})
                    existing_rects.append(new_rect)
                    break
        return targets

    def _add_glow_effect(self, pos, color):
        self.effects.append(GlowEffect(pos, color, self.np_random))

    def close(self):
        pygame.quit()

class GlowEffect:
    def __init__(self, pos, color, np_random):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.max_radius = np_random.integers(20, 40)
        self.lifespan = 15 # frames
        self.age = 0
        self.radius = 0
        self.alpha = 255

    def update(self):
        self.age += 1
        if self.age >= self.lifespan:
            return False
        
        progress = self.age / self.lifespan
        self.radius = int(self.max_radius * math.sin(progress * math.pi)) # Easing out
        self.alpha = int(255 * (1 - progress))
        return True

    def draw(self, surface):
        if self.radius > 0:
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                temp_surf, self.radius, self.radius, self.radius,
                (*self.color, self.alpha // 3) # Inner fill
            )
            pygame.gfxdraw.aacircle(
                temp_surf, self.radius, self.radius, self.radius,
                (*self.color, self.alpha) # Outer antialiased circle
            )
            surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))

if __name__ == '__main__':
    # --- Example Usage ---
    # Set a non-dummy driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    
    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Pixel Collector")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while not done:
        # --- Human Input Mapping ---
        movement_action = 0 # 0=none
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        # The MultiDiscrete action space is [movement, space, shift]
        action = [movement_action, 0, 0]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Colors: {info['colors_collected']}")

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            obs, info = env.reset()

        # --- Render to the screen ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()