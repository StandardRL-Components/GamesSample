import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:12:40.605591
# Source Brief: brief_00874.md
# Brief Index: 874
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "A physics-based puzzle game where you use elemental powers to guide an object to an exit. "
        "Manipulate the environment with fire, water, earth, and air to clear paths and push the goal."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press shift to cycle through elements and space to summon the selected element."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Game Design Constants ---
        self.GRID_SIZE = 20
        self.CURSOR_SPEED = self.GRID_SIZE
        self.MAX_GAME_SECONDS = 120
        self.MAX_STEPS = 3600 # 120s * 30fps

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_WALL = (80, 90, 110)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_GOAL = (255, 200, 50)
        self.COLOR_OBSTACLE = (200, 80, 80)
        self.COLOR_EARTH = (140, 100, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        
        # Elements
        self.ELEMENTS = ["fire", "water", "earth", "air"]
        self.ELEMENT_COLORS = {
            "fire": (255, 100, 0),
            "water": (0, 150, 255),
            "earth": self.COLOR_EARTH,
            "air": (200, 200, 255)
        }
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.time_left = None
        self.current_level = None
        
        self.cursor_pos = None
        self.goal_object = None
        self.level_exit = None
        
        self.element_counts = None
        self.selected_element_idx = None
        
        self.walls = None
        self.obstacles = None
        self.earth_blocks = None
        
        self.particles = None
        self.water_sources = None
        self.air_gusts = None
        
        self.prev_space_held = None
        self.prev_shift_held = None
        
        # Human rendering setup
        self.human_screen = None

        # The reset method is called implicitly by the environment wrapper,
        # but we can call it here to ensure state is initialized.
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.MAX_GAME_SECONDS
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        
        self._load_level(1)
        
        return self._get_observation(), self._get_info()

    def _load_level(self, level_num):
        self.current_level = level_num
        
        self.element_counts = {"fire": 5, "water": 5, "earth": 5, "air": 5}
        self.selected_element_idx = 0
        
        self.walls = []
        self.obstacles = []
        self.earth_blocks = []
        self.water_sources = []
        self.air_gusts = []
        
        level_data = self._get_level_data(level_num)
        
        for y, row in enumerate(level_data):
            for x, char in enumerate(row):
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                if char == '#':
                    self.walls.append(rect)
                elif char == 'S':
                    self.cursor_pos = pygame.Vector2(rect.center)
                elif char == 'G':
                    self.goal_object = pygame.Rect(rect)
                elif char == 'E':
                    self.level_exit = pygame.Rect(rect)
                elif char == 'O':
                    self.obstacles.append(rect)
        
        if self.cursor_pos is None: # Default cursor pos
            self.cursor_pos = pygame.Vector2(self.width // 2, self.height // 2)

    def _get_level_data(self, level_num):
        if level_num == 1:
            return [
                "################################",
                "#S                             #",
                "#                              #",
                "#                              #",
                "#         O                    #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#      G                       #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                        E     #",
                "#                              #",
                "################################",
            ]
        elif level_num == 2:
            return [
                "################################",
                "#S                             #",
                "#                              #",
                "#      ##################      #",
                "#      #                #      #",
                "#      # G              #      #",
                "#      #                #      #",
                "#      ##################      #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                        E     #",
                "#                              #",
                "################################",
            ]
        elif level_num == 3:
            return [
                "################################",
                "#S                             #",
                "#                              #",
                "#     #####                    #",
                "#     #                        #",
                "#     #   G                    #",
                "#     #                        #",
                "#     #                        #",
                "#     #                        #",
                "#     ##################       #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                              #",
                "#                        E     #",
                "#                              #",
                "#                              #",
                "################################",
            ]
        return self._get_level_data(1) # Default to level 1 if invalid

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action & Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        reward_info = {"obstacle_cleared": 0, "level_cleared": False}
        prev_goal_dist = self._get_goal_dist()

        self._handle_input(movement, space_pressed, shift_pressed, reward_info)
        
        # --- 2. Update Game State ---
        self._update_game_state()
        
        # --- 3. Check for Level Completion ---
        if self.level_exit and self.goal_object and self.level_exit.colliderect(self.goal_object):
            # SFX: Level Complete
            self.score += 10
            reward_info["level_cleared"] = True
            if self.current_level < 3:
                self._load_level(self.current_level + 1)
            else:
                self.win = True
                self.game_over = True
        
        # --- 4. Update Time & Check for Termination ---
        self.steps += 1
        self.time_left -= 1.0 / self.metadata["render_fps"]
        terminated = self._check_termination()

        # --- 5. Calculate Reward ---
        reward = self._calculate_reward(reward_info, prev_goal_dist)
        self.score += reward

        # --- 6. Return Tuple ---
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed, reward_info):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, self.GRID_SIZE/2, self.width - self.GRID_SIZE/2)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, self.GRID_SIZE/2, self.height - self.GRID_SIZE/2)

        # Cycle element
        if shift_pressed:
            # SFX: Element Switch
            self.selected_element_idx = (self.selected_element_idx + 1) % len(self.ELEMENTS)

        # Summon element
        if space_pressed:
            element_type = self.ELEMENTS[self.selected_element_idx]
            if self.element_counts[element_type] > 0:
                self.element_counts[element_type] -= 1
                self._summon_element(element_type, self.cursor_pos, reward_info)

    def _summon_element(self, element_type, pos, reward_info):
        grid_pos = pygame.Rect(
            int(pos.x // self.GRID_SIZE) * self.GRID_SIZE,
            int(pos.y // self.GRID_SIZE) * self.GRID_SIZE,
            self.GRID_SIZE, self.GRID_SIZE
        )

        if element_type == "fire":
            # SFX: Fire Whoosh
            for i in range(30):
                self.particles.append(Particle(grid_pos.center, "fire", self.np_random))
            for obstacle in self.obstacles[:]:
                if obstacle.colliderect(grid_pos):
                    self.obstacles.remove(obstacle)
                    reward_info["obstacle_cleared"] += 1
                    
        elif element_type == "water":
            # SFX: Water Splash
            if not any(grid_pos.colliderect(w) for w in self.walls + self.earth_blocks):
                self.water_sources.append({"pos": grid_pos.center, "life": 100})

        elif element_type == "earth":
            # SFX: Earth Rumble
            if not any(grid_pos.colliderect(o) for o in self.walls + self.obstacles + [self.goal_object, self.level_exit]):
                self.earth_blocks.append(grid_pos)

        elif element_type == "air":
            # SFX: Air Gust
            if not any(grid_pos.colliderect(w) for w in self.walls + self.earth_blocks):
                self.air_gusts.append({"rect": grid_pos, "life": 30})
                for i in range(20):
                    self.particles.append(Particle(grid_pos.center, "air", self.np_random))

    def _update_game_state(self):
        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

        # Update water
        for source in self.water_sources[:]:
            source["life"] -= 1
            if source["life"] <= 0:
                self.water_sources.remove(source)
            else:
                self.particles.append(Particle(source["pos"], "water", self.np_random))

        # Update air
        for gust in self.air_gusts[:]:
            gust["life"] -= 1
            if gust["life"] <= 0:
                self.air_gusts.remove(gust)
            else:
                if self.goal_object and gust["rect"].colliderect(self.goal_object):
                    direction = pygame.Vector2(self.goal_object.center) - pygame.Vector2(gust["rect"].center)
                    if direction.length() > 0:
                        move = direction.normalize() * 2
                        self._move_goal(move.x, move.y)

        # Update goal object based on water particles
        water_push = pygame.Vector2(0, 0)
        for p in self.particles:
            if p.type == "water" and self.goal_object and p.rect.colliderect(self.goal_object):
                water_push += p.vel
        if water_push.length() > 0:
            self._move_goal(water_push.x, water_push.y)

        # Anti-softlock: restart level if out of useful elements
        if sum(self.element_counts.values()) == 0 and not self.game_over:
            if not any(p.type in ["water", "air"] for p in self.particles) and not self.water_sources and not self.air_gusts:
                self._load_level(self.current_level)
                self.score -= 20 # Penalty for reset

    def _move_goal(self, dx, dy):
        if not self.goal_object: return
        self.goal_object.x += dx
        colliders = self.walls + self.obstacles + self.earth_blocks
        for c in colliders:
            if c.colliderect(self.goal_object):
                if dx > 0: self.goal_object.right = c.left
                if dx < 0: self.goal_object.left = c.right
                break
        self.goal_object.y += dy
        for c in colliders:
            if c.colliderect(self.goal_object):
                if dy > 0: self.goal_object.bottom = c.top
                if dy < 0: self.goal_object.top = c.bottom
                break
        
        self.goal_object.left = np.clip(self.goal_object.left, 0, self.width - self.goal_object.width)
        self.goal_object.top = np.clip(self.goal_object.top, 0, self.height - self.goal_object.height)

    def _get_goal_dist(self):
        if self.goal_object and self.level_exit:
            return pygame.Vector2(self.goal_object.center).distance_to(self.level_exit.center)
        return 0

    def _calculate_reward(self, reward_info, prev_dist):
        reward = 0
        
        # Continuous reward for moving goal closer to exit
        current_dist = self._get_goal_dist()
        reward += (prev_dist - current_dist) * 0.01

        # Event-based rewards
        reward += reward_info["obstacle_cleared"] * 5.0
        if reward_info["level_cleared"]:
            reward += 10.0
            
        # Terminal rewards
        if self.game_over:
            if self.win:
                reward += 100.0
            else: # Timeout
                reward -= 100.0
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.time_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "level": self.current_level,
            "elements": self.element_counts,
            "win": self.win
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        obs = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
        if self.render_mode == "human":
            if self.human_screen is None:
                self.human_screen = pygame.display.set_mode((self.width, self.height))
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            self.human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return obs

    def _render_game(self):
        # Draw grid
        for x in range(0, self.width, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y))

        # Draw level elements
        if self.level_exit: pygame.draw.rect(self.screen, self.COLOR_EXIT, self.level_exit)
        for wall in self.walls: pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        for obstacle in self.obstacles: pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
        for block in self.earth_blocks: pygame.draw.rect(self.screen, self.COLOR_EARTH, block)
        if self.goal_object: pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_object)
        
        # Draw particles and effects
        for gust in self.air_gusts:
            alpha = int(255 * (gust['life'] / 30.0))
            s = pygame.Surface(gust['rect'].size, pygame.SRCALPHA)
            s.fill((200, 200, 255, max(0, alpha // 4)))
            self.screen.blit(s, gust['rect'].topleft)

        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor
        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cx, cy), self.GRID_SIZE // 2, 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - self.GRID_SIZE, cy), (cx + self.GRID_SIZE, cy), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - self.GRID_SIZE), (cx, cy + self.GRID_SIZE), 1)

    def _render_ui(self):
        # Timer
        timer_text = self.font_ui.render(f"TIME: {int(self.time_left):03}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))
        
        # Level
        level_text = self.font_level.render(f"LEVEL {self.current_level}", True, (255, 255, 255))
        self.screen.blit(level_text, (self.width // 2 - level_text.get_width() // 2, 5))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Elements
        for i, element in enumerate(self.ELEMENTS):
            color = self.ELEMENT_COLORS[element]
            x_pos = 10 + i * 110
            y_pos = self.height - 40
            
            # Draw highlight for selected element
            if i == self.selected_element_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), (x_pos-5, y_pos-5, 100, 35), 2, border_radius=5)
            
            # Icon and text
            pygame.draw.rect(self.screen, color, (x_pos, y_pos, 25, 25), border_radius=5)
            count_text = self.font_ui.render(f"{element.upper()}: {self.element_counts[element]}", True, (255, 255, 255))
            self.screen.blit(count_text, (x_pos + 30, y_pos + 2))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "TIME UP!"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_text = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_text, (self.width // 2 - msg_text.get_width() // 2, self.height // 2 - msg_text.get_height() // 2))

    def close(self):
        pygame.quit()


class Particle:
    def __init__(self, pos, p_type, np_random=None):
        # Use standard library random if no generator is passed
        self._random = random if np_random is None else np_random
        
        self.type = p_type
        self.pos = pygame.Vector2(pos)
        
        if hasattr(self._random, 'integers'): # Gymnasium's np_random generator
            self.life = self._random.integers(20, 41)
        else: # Standard library random
            self.life = self._random.randint(20, 40)

        if self.type == "fire":
            angle = self._random.uniform(0, 2 * math.pi)
            speed = self._random.uniform(1, 4)
            self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            if hasattr(self._random, 'integers'):
                self.color = (self._random.integers(200, 256), self._random.integers(50, 151), 0)
                self.radius = self._random.integers(4, 9)
            else:
                self.color = (self._random.randint(200, 255), self._random.randint(50, 150), 0)
                self.radius = self._random.randint(4, 8)
        elif self.type == "water":
            self.vel = pygame.Vector2(self._random.uniform(-0.5, 0.5), self._random.uniform(2, 4))
            if hasattr(self._random, 'integers'):
                self.color = (self._random.integers(50, 101), self._random.integers(150, 201), 255)
                self.radius = self._random.integers(3, 7)
            else:
                self.color = (self._random.randint(50, 100), self._random.randint(150, 200), 255)
                self.radius = self._random.randint(3, 6)
        elif self.type == "air":
            angle = self._random.uniform(0, 2 * math.pi)
            speed = self._random.uniform(1, 3)
            self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.vel.x += self._random.uniform(-1, 1) # Swirl
            self.color = (220, 220, 255)
            if hasattr(self._random, 'integers'):
                self.radius = self._random.integers(2, 6)
            else:
                self.radius = self._random.randint(2, 5)

        self.rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)

    def update(self):
        self.life -= 1
        self.pos += self.vel
        if self.type == "fire":
            self.vel *= 0.95
            self.radius *= 0.97
        elif self.type == "water":
            self.vel.y += 0.1 # Gravity
        
        self.rect.center = self.pos

    def is_dead(self):
        return self.life <= 0 or self.radius < 1

    def draw(self, screen):
        alpha = int(255 * (self.life / 40.0))
        color_with_alpha = self.color + (max(0, alpha),)
        
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color_with_alpha, (self.radius, self.radius), int(self.radius))
        screen.blit(temp_surf, (self.pos.x - self.radius, self.pos.y - self.radius))


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()