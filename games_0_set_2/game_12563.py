import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a block stacking game.
    The goal is to stack falling blocks to reach a target height.
    Gravity increases with each block caught, making the stack less stable.
    Chain reactions can be triggered for score bonuses.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks to reach the target height. As you stack, gravity increases, "
        "making the tower less stable. Trigger chain reactions for score bonuses."
    )
    user_guide = "Use the ← and → arrow keys to move the catcher and stack the falling blocks."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TARGET_HEIGHT = 50
    FPS = 30
    TOTAL_TIME_SECONDS = 120
    MAX_STEPS = TOTAL_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_GRID = (30, 40, 60)
    COLOR_CATCHER = (0, 255, 255)
    COLOR_CATCHER_GLOW = (100, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_TARGET_LINE = (255, 50, 50)
    BLOCK_COLORS = {
        'normal': (0, 150, 255),
        'heavy': (200, 0, 255),
        'fragile': (255, 255, 0),
    }

    # Physics & Gameplay
    CATCHER_SPEED = 12
    GRAVITY_ACCEL = 0.25
    GRAVITY_MULTIPLIER_INCREMENT = 0.1
    BLOCK_BASE_WIDTH = 40
    BLOCK_BASE_HEIGHT = 15
    CHAIN_REACTION_VEL_THRESHOLD = 8.0
    CHAIN_REACTION_RADIUS = 75
    CHAIN_REACTION_FORCE = 5.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables (will be initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_condition = None
        self.catcher = None
        self.falling_blocks = None
        self.stacked_blocks = None
        self.particles = None
        self.gravity_multiplier = None
        self.reward_this_step = None
        self.had_blocks_in_stack = None
        self.collapse_detected = None

        # self.reset() is not called here to avoid creating a window during headless init
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.collapse_detected = False
        self.had_blocks_in_stack = False

        self.catcher = pygame.Rect(
            self.WIDTH // 2 - 50, self.HEIGHT - 30, 100, 10
        )
        self.falling_blocks = []
        self.stacked_blocks = []
        self.particles = []
        
        self.gravity_multiplier = 1.0
        self.reward_this_step = 0

        self._spawn_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        # Check if the episode was already terminated
        terminated = self._check_termination()
        if terminated:
            # If already terminated, just return the final state
            reward = self._calculate_reward()
            return self._get_observation(), reward, True, False, self._get_info()

        # If not terminated, proceed with the step
        self._handle_input(action)
        self._update_game_state()
        self.steps += 1
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        # The episode is truncated if it reaches the max steps without winning or losing
        if truncated:
            terminated = True 

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic ---
    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.catcher.x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher.x += self.CATCHER_SPEED
        
        self.catcher.x = max(0, min(self.WIDTH - self.catcher.width, self.catcher.x))

    def _update_game_state(self):
        self._update_falling_blocks()
        self._update_stack_stability()
        self._update_particles()
        
        if len(self.stacked_blocks) > 0:
            self.had_blocks_in_stack = True
        self._check_for_collapse()

        if not self.falling_blocks:
            self._spawn_block()
    
    def _spawn_block(self):
        stack_height = len(self.stacked_blocks)
        if stack_height < 5:
            block_type = 'normal'
        else:
            block_type = self.np_random.choice(['normal', 'heavy', 'fragile'], p=[0.6, 0.2, 0.2])

        width = self.BLOCK_BASE_WIDTH
        if block_type == 'heavy':
            width *= 1.2
        elif block_type == 'fragile':
            width *= 0.8
        
        x_pos = self.np_random.uniform(0, self.WIDTH - width)
        
        block = {
            "rect": pygame.Rect(x_pos, -self.BLOCK_BASE_HEIGHT, width, self.BLOCK_BASE_HEIGHT),
            "vel": pygame.Vector2(0, 0),
            "type": block_type,
            "color": self.BLOCK_COLORS[block_type],
            "stable": False
        }
        self.falling_blocks.append(block)

    def _update_falling_blocks(self):
        for block in self.falling_blocks[:]:
            block["vel"].y += self.GRAVITY_ACCEL * self.gravity_multiplier
            block["rect"].y += block["vel"].y
            block["rect"].x += block["vel"].x
            
            # Remove block if it falls off screen
            if block["rect"].top > self.HEIGHT:
                self.falling_blocks.remove(block)
                continue

            # Collision with catcher
            if block["rect"].colliderect(self.catcher) and block["vel"].y > 0:
                self._land_block(block, self.catcher)
                continue
            
            # Collision with stacked blocks
            for stacked_block in reversed(self.stacked_blocks):
                if block["rect"].colliderect(stacked_block["rect"]) and block["vel"].y > 0:
                    # Check for chain reaction
                    if block["vel"].y > self.CHAIN_REACTION_VEL_THRESHOLD or stacked_block["type"] == 'fragile':
                        self._trigger_chain_reaction(block["rect"].midbottom, block["vel"].y)
                        if block["type"] == 'fragile':
                            # Fragile blocks shatter and don't land
                            self.falling_blocks.remove(block)
                        else:
                            self._land_block(block, stacked_block["rect"])
                    else:
                        self._land_block(block, stacked_block["rect"])
                    break

    def _land_block(self, block, surface_rect):
        # Sound placeholder: # sfx_land_block.play()
        block["vel"] = pygame.Vector2(0, 0)
        block["rect"].bottom = surface_rect.top
        block["stable"] = True
        if block in self.falling_blocks:
            self.falling_blocks.remove(block)
        self.stacked_blocks.append(block)

        self.score += 1
        self.reward_this_step += 0.1
        self.gravity_multiplier += self.GRAVITY_MULTIPLIER_INCREMENT
    
    def _update_stack_stability(self):
        if not self.stacked_blocks:
            return

        is_stable = [False] * len(self.stacked_blocks)
        
        # The first block is stable if it's on the catcher
        base_block = self.stacked_blocks[0]
        support_rect = self.catcher
        if base_block["rect"].colliderect(support_rect):
            stability_factor = max(0.2, 1.0 - (self.gravity_multiplier - 1.0) * 0.2)
            allowed_offset = (support_rect.width / 2) * stability_factor
            if abs(base_block["rect"].centerx - support_rect.centerx) < allowed_offset:
                is_stable[0] = True

        # Check subsequent blocks
        for i in range(1, len(self.stacked_blocks)):
            block_above = self.stacked_blocks[i]
            block_below = self.stacked_blocks[i-1]
            
            if is_stable[i-1]: # Can only be stable if block below is stable
                stability_factor = max(0.2, 1.0 - (self.gravity_multiplier - 1.0) * 0.2)
                allowed_offset = (block_below["rect"].width / 2) * stability_factor
                if abs(block_above["rect"].centerx - block_below["rect"].centerx) < allowed_offset:
                    is_stable[i] = True

        # Topple unstable blocks
        for i in range(len(self.stacked_blocks) - 1, -1, -1):
            if not is_stable[i]:
                block = self.stacked_blocks.pop(i)
                block["stable"] = False
                # Give it a slight nudge
                nudge = (block["rect"].centerx - self.catcher.centerx) / self.catcher.width
                block["vel"].x = nudge * 2
                block["vel"].y = self.np_random.uniform(0, 1)
                self.falling_blocks.append(block)

    def _trigger_chain_reaction(self, pos, velocity):
        # Sound placeholder: # sfx_chain_reaction.play()
        num_particles = int(min(50, 10 + velocity * 2))
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": self.BLOCK_COLORS[self.np_random.choice(list(self.BLOCK_COLORS.keys()))],
                "radius": self.np_random.uniform(2, 5)
            })
        
        affected_blocks = 0
        for block in self.stacked_blocks[:]:
            dist = pygame.Vector2(block["rect"].center).distance_to(pos)
            if dist < self.CHAIN_REACTION_RADIUS:
                affected_blocks += 1
                block["stable"] = False # Mark for toppling
                push_vec = (pygame.Vector2(block["rect"].center) - pos)
                if push_vec.length() > 0:
                    force = self.CHAIN_REACTION_FORCE * (1 - dist / self.CHAIN_REACTION_RADIUS)
                    block["vel"] += push_vec.normalize() * force
        
        self.score += affected_blocks * 5
        self.reward_this_step += 1.0 * affected_blocks

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] *= 0.95
            if p["lifespan"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _check_for_collapse(self):
        if self.had_blocks_in_stack and not self.stacked_blocks:
            self.collapse_detected = True

    # --- Reward and Termination ---
    def _calculate_reward(self):
        reward = self.reward_this_step
        if self._check_termination():
            if self.win_condition:
                reward += 100
            elif self.collapse_detected:
                reward -= 100
        return reward

    def _check_termination(self):
        stack_height = len(self.stacked_blocks)
        if stack_height >= self.TARGET_HEIGHT:
            self.win_condition = True
            return True
        if self.collapse_detected:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    # --- Rendering ---
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Target Line
        target_y = self.HEIGHT - 30 - (self.TARGET_HEIGHT * self.BLOCK_BASE_HEIGHT)
        if target_y > 0:
            for x in range(0, self.WIDTH, 20):
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 2)

        # Catcher
        pygame.gfxdraw.box(self.screen, self.catcher, self.COLOR_CATCHER_GLOW)
        inner_catcher = self.catcher.inflate(-4, -4)
        pygame.gfxdraw.box(self.screen, inner_catcher, self.COLOR_CATCHER)

        # Blocks
        for block in self.stacked_blocks:
            self._draw_block(block["rect"], block["color"])
        for block in self.falling_blocks:
            self._draw_block(block["rect"], block["color"])
        
        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"]
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"]
            )

    def _draw_block(self, rect, color):
        pygame.gfxdraw.box(self.screen, rect, color)
        # 3D bevel effect
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, dark_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, dark_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        stack_height = len(self.stacked_blocks)
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        
        # Draw UI text with shadows
        self._draw_text(f"SCORE: {self.score}", (10, 10), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(f"HEIGHT: {stack_height}/{self.TARGET_HEIGHT}", (220, 10), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(f"TIME: {time_left:.1f}", (420, 10), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(f"GRAVITY: {self.gravity_multiplier:.1f}x", (520, 10), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if self._check_termination():
            if self.win_condition:
                msg = "LEVEL COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2 - 20), self.game_over_font, color, self.COLOR_TEXT_SHADOW, center=True)

    def _draw_text(self, text, pos, font, color, shadow_color, center=False):
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)

        shadow_rect = shadow_surf.get_rect()
        text_rect = text_surf.get_rect()

        if center:
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            text_rect.center = pos
        else:
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
            text_rect.topleft = pos

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    # --- Gymnasium Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": len(self.stacked_blocks),
            "gravity": self.gravity_multiplier,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display window
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    
    total_reward = 0
    clock = pygame.time.Clock()
    
    while running:
        # Player input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Movement, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait for 'r' to reset
            
    env.close()