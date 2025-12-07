import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# The following import is needed for the glowing circle effect
import pygame.gfxdraw

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a maze by changing shape between a square and a circle to pass through different gaps and reach the goal before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press Space to become a circle, and hold Shift to become a square."
    )
    auto_advance = True


    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (60, 70, 90)
    COLOR_GOAL = (80, 255, 120)
    COLOR_GOAL_GLOW = (180, 255, 200, 50)
    COLOR_TEXT = (230, 230, 240)

    COLOR_SQUARE = (80, 150, 255)
    COLOR_SQUARE_GLOW = (120, 180, 255, 60)
    COLOR_CIRCLE = (255, 100, 100)
    COLOR_CIRCLE_GLOW = (255, 150, 150, 60)

    CHAIN_COLORS = {
        "blue": (0, 120, 255),
        "red": (255, 0, 120),
    }
    PARTICLE_COLOR = (255, 220, 50)

    # Player settings
    PLAYER_SPEED = 4.0
    PLAYER_SIZE_SQUARE = 24
    PLAYER_SIZE_CIRCLE = 12 # Radius

    # Game settings
    MAX_EPISODE_STEPS = 1800 # 30 seconds at 60 FPS
    GAME_TIMER_SECONDS = 30.0

    # Reward settings
    REWARD_GOAL = 100.0
    REWARD_TIMEOUT = -10.0
    REWARD_CHAIN_REACTION = 5.0
    REWARD_STEP_PENALTY = -0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.player_pos = None
        self.player_shape = None
        self.walls = []
        self.goal_rect = None
        self.chain_blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.player_vel = pygame.math.Vector2(0, 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_TIMER_SECONDS
        self.player_shape = 'square'
        self.particles = []

        self._create_maze_layout()
        
        return self._get_observation(), self._get_info()

    def _create_maze_layout(self):
        self.walls = []
        self.chain_blocks = []
        
        # Maze wall definitions (x, y, w, h)
        wall_definitions = [
            (0, 0, self.SCREEN_WIDTH, 10), (0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10),
            (0, 0, 10, self.SCREEN_HEIGHT), (self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT),
            (150, 10, 10, 200), (150, 250, 10, 140), # Vertical wall with circle gap
            (150, 10, 250, 10), (300, 10, 10, 100),
            (300, 150, 10, 240), # Vertical wall with square gap
            (300, 300, 200, 10), (450, 150, 10, 150),
            (450, 10, 10, 100)
        ]
        for w in wall_definitions:
            self.walls.append(pygame.Rect(w))

        # Place player
        self.player_pos = pygame.math.Vector2(50, 50)

        # Place goal
        self.goal_rect = pygame.Rect(550, 340, 40, 40)

        # Place chain reaction blocks
        chain_block_size = 20
        chain_block_defs = [
            # Blue cluster
            (200, 100, "blue"), (220, 100, "blue"), (200, 120, "blue"),
            # Red cluster
            (350, 50, "red"), (370, 50, "red"), (350, 70, "red"), (370, 70, "red"),
        ]
        for x, y, color_key in chain_block_defs:
            self.chain_blocks.append({
                "rect": pygame.Rect(x, y, chain_block_size, chain_block_size),
                "color_key": color_key,
                "color_val": self.CHAIN_COLORS[color_key],
                "active": True
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self.REWARD_STEP_PENALTY
        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # 1. Handle actions (shape change first)
        self._handle_actions(action)

        # 2. Update game state and handle collisions
        reward += self._update_player_position()
        
        # 3. Update particles
        self._update_particles()
        
        # 4. Check for termination conditions
        terminated = False
        if self.player_rect.colliderect(self.goal_rect):
            reward += self.REWARD_GOAL
            self.score += self.REWARD_GOAL
            terminated = True
            self.game_over = True
        elif self.timer <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            reward += self.REWARD_TIMEOUT
            self.score += self.REWARD_TIMEOUT
            terminated = True
            self.game_over = True
        
        # The score is cumulative, so we add the step reward to it
        self.score += reward - self.REWARD_STEP_PENALTY
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Shape change has priority
        new_shape = self.player_shape
        if shift_held:
            new_shape = 'square'
        elif space_held:
            new_shape = 'circle'
        
        if new_shape != self.player_shape:
            self.player_shape = new_shape
            self._spawn_particles(self.player_pos, 5, self.player_color, 0.5)

        # Movement
        self.player_vel = pygame.math.Vector2(0, 0)
        if movement == 1: # Up
            self.player_vel.y = -self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_vel.y = self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED

    def _update_player_position(self):
        reward = 0
        new_pos = self.player_pos + self.player_vel
        
        # Create proposed new rect for collision checks
        if self.player_shape == 'square':
            new_rect = pygame.Rect(
                new_pos.x - self.PLAYER_SIZE_SQUARE / 2,
                new_pos.y - self.PLAYER_SIZE_SQUARE / 2,
                self.PLAYER_SIZE_SQUARE, self.PLAYER_SIZE_SQUARE
            )
        else: # Circle
            new_rect = pygame.Rect(
                new_pos.x - self.PLAYER_SIZE_CIRCLE,
                new_pos.y - self.PLAYER_SIZE_CIRCLE,
                self.PLAYER_SIZE_CIRCLE * 2, self.PLAYER_SIZE_CIRCLE * 2
            )

        # Wall collision
        wall_collision = any(new_rect.colliderect(wall) for wall in self.walls)
        
        # Shape-based wall collision
        if self.player_shape == 'square':
             # Squares can't pass through circle gaps (width/height < 25)
            if any(new_rect.colliderect(w) for w in self.walls if w.width < 25 or w.height < 25):
                 wall_collision = True
        
        if not wall_collision:
            self.player_pos = new_pos

        # Keep player in bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

        # Chain block collision
        player_r = self.player_rect
        for block in self.chain_blocks:
            if block["active"] and player_r.colliderect(block["rect"]):
                block["active"] = False
                reward += self.REWARD_CHAIN_REACTION
                self._spawn_particles(pygame.math.Vector2(block["rect"].center), 20, self.PARTICLE_COLOR, 1.0)
                self._trigger_chain_reaction(block)
        
        return reward

    def _trigger_chain_reaction(self, origin_block):
        q = [origin_block]
        visited = {id(origin_block)}

        while q:
            current_block = q.pop(0)
            for other_block in self.chain_blocks:
                if other_block["active"] and id(other_block) not in visited:
                    if other_block["color_key"] == current_block["color_key"]:
                        # Check for adjacency (touching)
                        if current_block["rect"].colliderect(other_block["rect"].inflate(2, 2)):
                            other_block["active"] = False
                            self._spawn_particles(pygame.math.Vector2(other_block["rect"].center), 10, self.PARTICLE_COLOR, 0.8)
                            visited.add(id(other_block))
                            q.append(other_block)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw goal
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect, border_radius=8)
        self._draw_glowing_rect(self.goal_rect, self.COLOR_GOAL_GLOW, 15, 8)

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw chain blocks
        for block in self.chain_blocks:
            if block["active"]:
                pygame.draw.rect(self.screen, block["color_val"], block["rect"], border_radius=4)
        
        # Draw particles
        self._draw_particles()

        # Draw player
        self._draw_player()

    def _draw_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        if self.player_shape == 'square':
            size = self.PLAYER_SIZE_SQUARE
            rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            self._draw_glowing_rect(rect, self.COLOR_SQUARE_GLOW, 20, 6)
            pygame.draw.rect(self.screen, self.COLOR_SQUARE, rect, border_radius=6)
        else: # Circle
            radius = self.PLAYER_SIZE_CIRCLE
            self._draw_glowing_circle(pos, radius, self.COLOR_CIRCLE_GLOW, 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CIRCLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CIRCLE)

    def _draw_glowing_rect(self, rect, color, glow_size, border_radius):
        glow_rect = rect.inflate(glow_size, glow_size)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=border_radius + glow_size//2)
        self.screen.blit(shape_surf, glow_rect.topleft)

    def _draw_glowing_circle(self, pos, radius, color, glow_size):
        glow_radius = radius + glow_size // 2
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _spawn_particles(self, pos, count, color, max_lifespan):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": velocity,
                "lifespan": random.uniform(0.5, max_lifespan),
                "max_lifespan": max_lifespan,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["lifespan"] -= 1 / self.FPS
            if p["lifespan"] <= 0:
                self.particles.remove(p)
    
    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"] + (alpha,)
            radius = int(3 * (p["lifespan"] / p["max_lifespan"]))
            if radius > 0:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (int(p["pos"].x-radius), int(p["pos"].y-radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {self.timer:.2f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 15, 15))

        # Score
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 15))

        # Shape indicator
        shape_text = f"SHAPE: {self.player_shape.upper()}"
        shape_surf = self.font_small.render(shape_text, True, self.player_color)
        self.screen.blit(shape_surf, (15, self.SCREEN_HEIGHT - shape_surf.get_height() - 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_shape": self.player_shape,
            "player_pos": (self.player_pos.x, self.player_pos.y),
        }

    @property
    def player_rect(self):
        if self.player_shape == 'square':
            size = self.PLAYER_SIZE_SQUARE
            return pygame.Rect(self.player_pos.x - size/2, self.player_pos.y - size/2, size, size)
        else:
            radius = self.PLAYER_SIZE_CIRCLE
            return pygame.Rect(self.player_pos.x - radius, self.player_pos.y - radius, radius*2, radius*2)
            
    @property
    def player_color(self):
        return self.COLOR_SQUARE if self.player_shape == 'square' else self.COLOR_CIRCLE

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play ---
    # To run in a window, comment out the `os.environ` line at the top
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0
        shift_held = 0

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
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0.0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(GameEnv.FPS)

    env.close()