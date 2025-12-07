import gymnasium as gym
import os
import pygame
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Block:
    """A helper class to store block properties."""
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.mass = width * height  # Assuming uniform density

    def draw(self, surface, camera_offset_y=0):
        """Draws the block with a border for better visibility."""
        draw_rect = self.rect.move(0, camera_offset_y)
        pygame.draw.rect(surface, self.color, draw_rect)
        pygame.draw.rect(surface, (0, 0, 0, 100), draw_rect, 1)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack blocks to build the tallest tower possible. Place blocks carefully, as an unstable tower will collapse!"
    )
    user_guide = (
        "Use arrow keys to select a drop zone and press space to place the block."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.VICTORY_BLOCKS = 25
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = pygame.Color(210, 220, 230)
        self.COLOR_GROUND = pygame.Color(115, 90, 70)
        self.COLOR_BLOCK_START = pygame.Color(144, 238, 144) # Light Green
        self.COLOR_BLOCK_END = pygame.Color(0, 100, 0) # Dark Green
        self.COLOR_UI_TEXT = pygame.Color(30, 30, 30)
        self.COLOR_PREVIEW = pygame.Color(255, 255, 255, 128)

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks = []
        self.base_block = None
        self.hover_block = None
        self.particles = []
        self.target_x_pos = self.WIDTH / 2
        self.camera_y_offset = 0.0
        self.current_block_size_multiplier = 1.0
        self.placement_range_multiplier = 1.0
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_y_offset = 0.0
        self.current_block_size_multiplier = 1.0
        self.placement_range_multiplier = 1.0
        
        # Create the static base block
        base_width = self.WIDTH * 0.4
        base_height = 20
        base_x = (self.WIDTH - base_width) / 2
        base_y = self.HEIGHT - 40
        self.base_block = Block(base_x, base_y, base_width, base_height, self.COLOR_GROUND)
        self.blocks = [self.base_block]

        # Initialize the first hovering block
        self.target_x_pos = self.WIDTH / 2
        self._prepare_next_block()

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        movement, place_block, _ = action
        place_block_pressed = (place_block == 1)

        # 1. Handle Movement (Update Target Position)
        if movement > 0:
            self._update_target_position(movement)
        
        # 2. Handle Block Placement
        if place_block_pressed:
            placed_block = self._place_block()
            self._create_placement_particles(placed_block.rect.midbottom)
            
            if self._check_collapse():
                self.game_over = True
                terminated = True
                reward = -100
                self._create_collapse_particles()
            else:
                reward = 1.0
                self.score += 1
                
                if self.score >= self.VICTORY_BLOCKS:
                    self.game_over = True
                    terminated = True
                    reward += 100
                else:
                    self._prepare_next_block()
        
        # 3. Update game systems
        self._update_particles()
        self._update_camera()

        # 4. Check for termination by max steps
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_target_position(self, movement_action):
        """Maps movement action (1-4) to a horizontal position."""
        effective_width = (self.WIDTH * 0.8) * self.placement_range_multiplier
        min_x = (self.WIDTH - effective_width) / 2
        
        num_segments = 4
        segment_width = effective_width / num_segments
        
        segment_index = movement_action - 1
        self.target_x_pos = min_x + (segment_index * segment_width) + (segment_width / 2)

    def _place_block(self):
        """Creates a new block and adds it to the stack."""
        new_block = self.hover_block
        new_block.rect.centerx = int(self.target_x_pos)
        self.blocks.append(new_block)
        return new_block
    
    def _prepare_next_block(self):
        """Updates progression and creates the next hovering block."""
        self.current_block_size_multiplier *= 0.99
        if self.score > 0 and self.score % 10 == 0:
            self.placement_range_multiplier *= 0.99
        
        top_block = self.blocks[-1]
        
        width = int(100 * self.current_block_size_multiplier)
        height = 20
        x = self.target_x_pos - width / 2
        y = top_block.rect.top - height

        progress = min(1.0, self.score / self.VICTORY_BLOCKS)
        color = self.COLOR_BLOCK_START.lerp(self.COLOR_BLOCK_END, progress)

        self.hover_block = Block(x, y, width, height, color)

    def _check_collapse(self):
        """Checks if the tower is unstable."""
        if len(self.blocks) <= 2:
            return False

        for i in range(len(self.blocks) - 2, 0, -1):
            supporting_block = self.blocks[i]
            
            total_mass = 0
            weighted_x_sum = 0
            for j in range(i + 1, len(self.blocks)):
                block_above = self.blocks[j]
                total_mass += block_above.mass
                weighted_x_sum += block_above.rect.centerx * block_above.mass
            
            if total_mass == 0: continue

            com_x = weighted_x_sum / total_mass
            
            if not (supporting_block.rect.left <= com_x <= supporting_block.rect.right):
                return True
        
        return False

    def _update_camera(self):
        """Adjusts camera to keep the top of the tower in view."""
        if not self.blocks: return

        top_y = self.blocks[-1].rect.top
        camera_trigger_y = self.HEIGHT * 0.4
        
        if top_y < camera_trigger_y:
            target_offset = camera_trigger_y - top_y
            self.camera_y_offset += (target_offset - self.camera_y_offset) * 0.1

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        surviving_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                surviving_particles.append(p)
        self.particles = surviving_particles

    def _create_placement_particles(self, pos):
        """Generates particles for a block placement effect."""
        for _ in range(15):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-1, 1), random.uniform(0, 1.5)],
                'life': random.randint(10, 20),
                'color': (180, 180, 180),
                'radius': random.uniform(1, 3)
            })

    def _create_collapse_particles(self):
        """Generates a particle explosion for tower collapse."""
        for block in self.blocks[1:]:
            for _ in range(30):
                self.particles.append({
                    'pos': list(block.rect.center),
                    'vel': [random.uniform(-4, 4), random.uniform(-5, 2)],
                    'life': random.randint(20, 40),
                    'color': block.color,
                    'radius': random.uniform(2, 5)
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all primary game elements."""
        offset = int(self.camera_y_offset)

        ground_rect = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        for block in self.blocks:
            block.draw(self.screen, offset)

        if not self.game_over and self.hover_block:
            self.hover_block.rect.centerx += (self.target_x_pos - self.hover_block.rect.centerx) * 0.25
            
            preview_rect = self.hover_block.rect.copy()
            preview_rect.move_ip(0, offset)
            
            preview_surface = pygame.Surface(preview_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(preview_surface, self.COLOR_PREVIEW, (0, 0, *preview_rect.size), border_radius=3)
            self.screen.blit(preview_surface, preview_rect.topleft)

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] + offset))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_ui(self):
        """Renders UI elements like score."""
        score_text = self.font.render(f"Blocks: {self.score}/{self.VICTORY_BLOCKS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            end_text_str = "VICTORY!" if self.score >= self.VICTORY_BLOCKS else "TOWER COLLAPSED"
            end_font = pygame.font.SysFont("Impact", 60, bold=True)
            end_text = end_font.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_placed": len(self.blocks) - 1
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1 or event.key == pygame.K_LEFT: action[0] = 1
                elif event.key == pygame.K_2: action[0] = 2
                elif event.key == pygame.K_3: action[0] = 3
                elif event.key == pygame.K_4 or event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                     running = False
        
        if action[0] != 0 or action[1] != 0:
             obs, reward, terminated, truncated, info = env.step(action)
             if terminated or truncated:
                 print(f"Game Over. Score: {info['score']}, Reward: {reward}")

        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()