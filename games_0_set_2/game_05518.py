
# Generated: 2025-08-28T05:15:51.120486
# Source Brief: brief_05518.md
# Brief Index: 5518

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the falling block. Press Space to drop it."

    # Must be a short, user-facing description of the game:
    game_description = "Stack falling blocks as high as you can. The tower will topple if it becomes unstable!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TARGET_HEIGHT = 20

        # Game element sizes and speeds
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 40, 20
        self.PLAYER_SPEED = 8
        self.INITIAL_FALL_SPEED = 2.0 # pixels per frame

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 70)
        self.COLOR_TARGET_LINE = (255, 255, 0, 100)
        self.BLOCK_COLORS = [
            (231, 76, 60),  # Red
            (46, 204, 113), # Green
            (52, 152, 219), # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
        ]
        self.COLOR_UI_TEXT = (240, 240, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = 0.0
        self.last_space_press = False
        self.np_random = None
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0 # Represents height
        self.game_over = False
        self.stacked_blocks = []
        self.particles = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_space_press = False
        
        self._spawn_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _spawn_block(self):
        x = self.WIDTH / 2 - self.BLOCK_WIDTH / 2
        y = -self.BLOCK_HEIGHT
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        color = self.BLOCK_COLORS[color_index]
        
        self.falling_block = {
            "rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color": color,
            "vy": self.fall_speed,
        }
    
    def step(self, action):
        reward = 0
        is_game_over_this_step = False

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 50 == 0:
            self.fall_speed += 0.1

        # Spawn a new block if the previous one landed
        if self.falling_block is None and not self.game_over:
            self._spawn_block()

        # Handle falling block logic
        if self.falling_block and not self.game_over:
            # 1. Handle player input
            if movement == 3:  # Left
                self.falling_block["rect"].x -= self.PLAYER_SPEED
            elif movement == 4:  # Right
                self.falling_block["rect"].x += self.PLAYER_SPEED

            self.falling_block["rect"].x = max(0, self.falling_block["rect"].x)
            self.falling_block["rect"].x = min(self.WIDTH - self.BLOCK_WIDTH, self.falling_block["rect"].x)

            # Drop action (triggered on press, not hold)
            if space_held and not self.last_space_press:
                target_y = self.HEIGHT - self.BLOCK_HEIGHT
                highest_block = self._get_highest_stacked_block(self.falling_block["rect"])
                if highest_block:
                    target_y = highest_block["rect"].top - self.BLOCK_HEIGHT
                self.falling_block["rect"].top = target_y
                # Sound effect placeholder: whoosh

            self.last_space_press = space_held

            # 2. Update vertical position
            self.falling_block["rect"].y += self.falling_block["vy"]

            # 3. Check for collision
            landed = False
            support_block = None
            if self.falling_block["rect"].bottom >= self.HEIGHT:
                self.falling_block["rect"].bottom = self.HEIGHT
                landed = True
            else:
                support_block = self._get_highest_stacked_block(self.falling_block["rect"])
                if support_block and self.falling_block["rect"].colliderect(support_block["rect"]):
                    self.falling_block["rect"].bottom = support_block["rect"].top
                    landed = True

            # 4. Handle landing
            if landed:
                # Sound effect placeholder: thud
                self._create_particles(self.falling_block["rect"].midbottom, self.falling_block["color"])
                self.stacked_blocks.append(self.falling_block)
                
                reward += 0.1  # Reward for placing a block
                
                new_height = len(self.stacked_blocks)
                if new_height > self.score:
                    self.score = new_height
                    reward += 1.0
                
                if support_block:
                    overhang = max(0, support_block['rect'].left - self.falling_block['rect'].left) + \
                               max(0, self.falling_block['rect'].right - support_block['rect'].right)
                    if overhang > self.BLOCK_WIDTH * 0.5:
                        reward -= 5.0
                
                if not self._check_stability():
                    is_game_over_this_step = True
                    self.game_over = True
                    reward = -100.0
                    # Sound effect placeholder: crash
                    self._create_collapse_particles()
                
                self.falling_block = None

        self._update_particles()
        self.steps += 1
        
        terminated = self.game_over
        if self.score >= self.TARGET_HEIGHT and not is_game_over_this_step:
            reward = 100.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_highest_stacked_block(self, rect):
        highest = None
        for block in self.stacked_blocks:
            if rect.colliderect(pygame.Rect(block["rect"].x, 0, block["rect"].width, self.HEIGHT)):
                if highest is None or block["rect"].top < highest["rect"].top:
                    highest = block
        return highest

    def _check_stability(self):
        if len(self.stacked_blocks) <= 1:
            return True
        
        base_block = self.stacked_blocks[0]
        base_support_min = base_block["rect"].left
        base_support_max = base_block["rect"].right

        total_mass_moment = sum(block["rect"].centerx for block in self.stacked_blocks)
        cog_x = total_mass_moment / len(self.stacked_blocks)

        return base_support_min <= cog_x <= base_support_max

    def _create_particles(self, pos, color, count=10):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color,
            })

    def _create_collapse_particles(self):
        for block in self.stacked_blocks:
            self._create_particles(block["rect"].center, block["color"], count=5)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
    
    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        target_y = self.HEIGHT - (self.TARGET_HEIGHT * self.BLOCK_HEIGHT)
        if target_y > 0:
            for x in range(0, self.WIDTH, 20):
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 1)

        for block in self.stacked_blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, (0,0,0,50), block["rect"], 1)

        if self.falling_block:
            pygame.draw.rect(self.screen, self.falling_block["color"], self.falling_block["rect"])
            pygame.draw.rect(self.screen, (255,255,255,150), self.falling_block["rect"], 2)

            target_y = self.HEIGHT - self.BLOCK_HEIGHT
            highest_block = self._get_highest_stacked_block(self.falling_block["rect"])
            if highest_block:
                target_y = highest_block["rect"].top - self.BLOCK_HEIGHT
            
            pred_rect = self.falling_block["rect"].copy()
            pred_rect.top = target_y
            
            s = pygame.Surface((self.BLOCK_WIDTH, self.BLOCK_HEIGHT), pygame.SRCALPHA)
            s.fill((255, 255, 255, 50))
            self.screen.blit(s, pred_rect.topleft)

        for p in self.particles:
            size = max(1, p["life"] / 4)
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(size))

    def _render_ui(self):
        score_text = self.font_large.render(f"Height: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (20, 50))
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            status_text = "TOWER COLLAPSED"
            if self.score >= self.TARGET_HEIGHT:
                status_text = "GOAL REACHED!"

            game_over_surf = self.font_large.render(status_text, True, (255, 80, 80))
            text_rect = game_over_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    import os
    # Set a dummy video driver to run headless if not in main
    if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
        env = GameEnv(render_mode="rgb_array")
        
        pygame.display.set_caption("Block Stacker")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            action[0] = 0
            action[1] = 0
            action[2] = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
            if keys[pygame.K_SPACE]:
                action[1] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
                
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)
            
        env.close()
        pygame.quit()