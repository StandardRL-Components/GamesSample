
# Generated: 2025-08-28T01:25:37.050732
# Source Brief: brief_04098.md
# Brief Index: 4098

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the catcher."

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruits and avoid bombs in a fast-paced arcade game. "
        "Catch 25 fruits to win, but catching 3 bombs ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 40
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)

        # Colors
        self.COLOR_BG = (135, 206, 235)  # Light Sky Blue
        self.COLOR_GRID = (110, 180, 210)
        self.COLOR_CATCHER = (139, 69, 19)  # Saddle Brown
        self.COLOR_CATCHER_BORDER = (92, 51, 23)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_SKULL = (220, 220, 220)
        self.FRUIT_COLORS = {
            "apple": (220, 20, 60),    # Crimson
            "orange": (255, 165, 0), # Orange
            "lemon": (255, 255, 0),   # Yellow
            "lime": (50, 205, 50),     # LimeGreen
            "grape": (138, 43, 226),  # BlueViolet
        }
        self.PARTICLE_COLORS = {
            "explosion": [(255, 69, 0), (255, 140, 0), (169, 169, 169)]
        }

        # Game parameters
        self.MAX_FRUITS = 25
        self.MAX_BOMBS = 3
        self.MAX_STEPS = 1000
        self.CATCHER_WIDTH = 80
        self.CATCHER_HEIGHT = 20
        self.CATCHER_SPEED = 10
        self.OBJECT_RADIUS = 15
        self.BASE_FALL_SPEED = 2.5
        self.SPAWN_PROBABILITY = 0.04
        self.BOMB_PROBABILITY = 0.25

        # Initialize state variables
        self.np_random = None
        self.catcher_pos_x = 0
        self.falling_objects = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.win = False
        self.fall_speed_multiplier = 1.0

        # Run validation check
        # self.validate_implementation() # Commented out for submission
        
        # Initialize state variables for the first time
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.catcher_pos_x = self.SCREEN_WIDTH // 2
        self.falling_objects = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.win = False
        self.fall_speed_multiplier = 1.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0] # 0-4: none/up/down/left/right

        # --- Update Game Logic ---
        self._handle_movement(movement)
        self._update_difficulty()
        self._spawn_objects()
        
        collision_reward = self._update_objects_and_collisions()
        reward += collision_reward

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self.win:
                reward += 50  # Goal-oriented win reward
            else: # Loss by bombs or steps
                reward -= 50  # Goal-oriented loss reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 3:  # Left
            self.catcher_pos_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_pos_x += self.CATCHER_SPEED
        
        # Clamp catcher position to screen bounds
        self.catcher_pos_x = max(
            self.CATCHER_WIDTH // 2, 
            min(self.catcher_pos_x, self.SCREEN_WIDTH - self.CATCHER_WIDTH // 2)
        )

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.fall_speed_multiplier += 0.1

    def _spawn_objects(self):
        if self.np_random.random() < self.SPAWN_PROBABILITY:
            spawn_x = self.np_random.integers(self.OBJECT_RADIUS, self.SCREEN_WIDTH - self.OBJECT_RADIUS)
            
            is_bomb = self.np_random.random() < self.BOMB_PROBABILITY
            if is_bomb:
                obj_type = "bomb"
                color = self.COLOR_BOMB
            else:
                obj_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
                color = self.FRUIT_COLORS[obj_type]
            
            self.falling_objects.append({
                "pos": [spawn_x, -self.OBJECT_RADIUS],
                "type": obj_type,
                "color": color
            })
    
    def _update_objects_and_collisions(self):
        step_reward = 0
        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.CATCHER_WIDTH // 2,
            self.SCREEN_HEIGHT - self.CATCHER_HEIGHT,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )

        for obj in reversed(self.falling_objects):
            # Move object
            obj["pos"][1] += self.BASE_FALL_SPEED * self.fall_speed_multiplier

            # Check for collision
            obj_rect = pygame.Rect(
                obj["pos"][0] - self.OBJECT_RADIUS,
                obj["pos"][1] - self.OBJECT_RADIUS,
                self.OBJECT_RADIUS * 2,
                self.OBJECT_RADIUS * 2
            )

            if catcher_rect.colliderect(obj_rect):
                if "bomb" in obj["type"]:
                    self.bombs_caught += 1
                    step_reward -= 5  # Penalty for catching a bomb
                    self._create_particles(obj["pos"], "explosion", 50)
                    # Sound: Explosion
                else: # Fruit
                    self.score += 1
                    step_reward += 1 # Reward for catching fruit

                    # Check for risky catch bonus
                    for other_obj in self.falling_objects:
                        if "bomb" in other_obj["type"]:
                            dist = math.hypot(obj["pos"][0] - other_obj["pos"][0], obj["pos"][1] - other_obj["pos"][1])
                            if dist < self.GRID_SIZE * 2.5:
                                step_reward += 5 # Bonus for risky catch
                                break # only one bonus per catch

                    self._create_particles(obj["pos"], obj["color"], 25)
                    # Sound: Fruit catch
                
                self.falling_objects.remove(obj)

            # Remove if off-screen
            elif obj["pos"][1] > self.SCREEN_HEIGHT + self.OBJECT_RADIUS:
                self.falling_objects.remove(obj)
        
        return step_reward

    def _create_particles(self, pos, color, count):
        is_explosion = color == "explosion"
        for _ in range(count):
            if is_explosion:
                p_color = self.np_random.choice(self.PARTICLE_COLORS["explosion"])
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(15, 30)
                radius = self.np_random.uniform(1, 4)
            else: # Fruit
                p_color = color
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                vel[1] -= 1 # Add a little upward pop
                life = self.np_random.integers(20, 40)
                radius = self.np_random.uniform(1, 3)

            self.particles.append({
                "pos": list(pos), "vel": vel, "life": life, "max_life": life, "color": p_color, "radius": radius
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.score >= self.MAX_FRUITS:
            self.win = True
            return True
        if self.bombs_caught >= self.MAX_BOMBS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Draw falling objects
        for obj in self.falling_objects:
            pos_x, pos_y = int(obj["pos"][0]), int(obj["pos"][1])
            if "bomb" in obj["type"]:
                self._draw_bomb_icon(self.screen, pos_x, pos_y, self.OBJECT_RADIUS)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.OBJECT_RADIUS, obj["color"])
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.OBJECT_RADIUS, obj["color"])

        # Draw catcher
        catcher_rect = (
            int(self.catcher_pos_x - self.CATCHER_WIDTH // 2),
            self.SCREEN_HEIGHT - self.CATCHER_HEIGHT,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        points = [
            (catcher_rect[0], catcher_rect[1] + self.CATCHER_HEIGHT),
            (catcher_rect[0] + self.CATCHER_WIDTH, catcher_rect[1] + self.CATCHER_HEIGHT),
            (catcher_rect[0] + self.CATCHER_WIDTH - 10, catcher_rect[1]),
            (catcher_rect[0] + 10, catcher_rect[1])
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CATCHER, points)
        pygame.draw.polygon(self.screen, self.COLOR_CATCHER_BORDER, points, 3)

    def _draw_bomb_icon(self, surface, center_x, center_y, radius):
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, self.COLOR_BOMB)
        
        # Skull
        eye_radius = int(radius * 0.2)
        eye_offset_x = int(radius * 0.4)
        eye_offset_y = int(radius * 0.2)
        pygame.draw.circle(surface, self.COLOR_SKULL, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius)
        pygame.draw.circle(surface, self.COLOR_SKULL, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius)
        
        nose_rect = pygame.Rect(center_x - 2, center_y, 4, 4)
        pygame.draw.rect(surface, self.COLOR_SKULL, nose_rect)

        mouth_y = center_y + int(radius * 0.4)
        for i in range(4):
            x_start = center_x - int(radius*0.5) + i * int(radius*0.3)
            pygame.draw.line(surface, self.COLOR_SKULL, (x_start, mouth_y), (x_start, mouth_y + 4), 2)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"FRUITS: {self.score}/{self.MAX_FRUITS}", True, (255, 255, 255))
        score_bg = score_text.get_rect(topleft=(8, 8))
        score_bg.inflate_ip(10, 5)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), score_bg, border_radius=5)
        self.screen.blit(score_text, (13, 10))

        # Bombs
        for i in range(self.MAX_BOMBS):
            bomb_x = self.SCREEN_WIDTH - 25 - (i * 35)
            bomb_y = 25
            if i < self.bombs_caught:
                self._draw_bomb_icon(self.screen, bomb_x, bomb_y, 12)
            else:
                pygame.gfxdraw.aacircle(self.screen, bomb_x, bomb_y, 12, (0,0,0, 64))
                pygame.gfxdraw.filled_circle(self.screen, bomb_x, bomb_y, 12, (0,0,0, 64))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (152, 251, 152) if self.win else (255, 99, 71)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_caught": self.bombs_caught,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # It will not be executed when the environment is used by an RL agent
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        # Human controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        action[0] = 0 # No movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()