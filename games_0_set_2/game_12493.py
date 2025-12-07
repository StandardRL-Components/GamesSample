import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:55.557306
# Source Brief: brief_02493.md
# Brief Index: 2493
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for a single code fragment
class Fragment:
    def __init__(self, x, y, width, height, color, glyph):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.glyph = glyph
        self.settled = False
        self.velocity_y = 0

# Helper class for a particle effect
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.radius = random.uniform(2, 5)
        self.life = random.randint(20, 40)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.radius -= 0.1
        self.life -= 1
        return self.life > 0 and self.radius > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Flip gravity to match and clear falling code fragments in this fast-paced, "
        "cyberpunk-themed puzzle game. Create combos to score high before the screen fills up."
    )
    user_guide = "Controls: Press space to flip gravity. Match fragments of the same color to clear them."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SCORE = 1000
    MAX_STEPS = 2000
    FRAGMENT_WIDTH = 25
    FRAGMENT_HEIGHT = 25
    
    # --- COLORS (Cyberpunk/Matrix Theme) ---
    COLOR_BG = (5, 10, 5)
    COLOR_GRID = (20, 50, 20)
    FRAGMENT_COLORS = [
        (0, 255, 128),  # Matrix Green
        (128, 0, 255),  # Cyber Purple
        (255, 0, 128),  # Neon Pink
        (0, 192, 255),  # Electric Blue
        (255, 255, 0),  # Glitch Yellow
    ]
    COLOR_UI_TEXT = (200, 255, 200)
    COLOR_FILL_LINE = (255, 50, 50, 100) # RGBA for transparency

    # --- GLYPHS for fragments ---
    GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$&*()[]{}<>?/%"

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_glyph = pygame.font.SysFont("lucidaconsole", 16)

        self.fragments = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = 1  # 1 for down, -1 for up
        self.fall_speed = 0.0
        self.spawn_rate = 0.0
        self.spawn_timer = 0.0
        self.last_space_held = False

        self.reset()
        # self.validate_implementation() # Removed for production code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity = 1
        
        self.fall_speed_base = 1.0
        self.spawn_rate_base = 1.0 / 30.0 # 1 per second at 30fps

        self.fragments.clear()
        self.particles.clear()
        
        self.spawn_timer = 0.0
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward_this_step = 0

        # --- 1. Handle Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_pressed:
            self.gravity *= -1
            # # Sound effect placeholder
            # pygame.mixer.Sound("sounds/gravity_flip.wav").play()
            # Unsettle all fragments to let them fall in the new direction
            for frag in self.fragments:
                frag.settled = False

        # --- 2. Update Game State ---
        self._update_difficulty()
        self._spawn_fragments()
        
        # --- 3. Main Game Loop (Chain Reactions) ---
        chain_reaction_iterations = 0
        while True:
            chain_reaction_iterations += 1
            if chain_reaction_iterations > 10: # Safety break
                break

            self._move_and_settle_fragments()
            
            cleared_info = self._find_and_clear_matches()
            num_cleared = cleared_info["count"]
            
            if num_cleared > 0:
                self.score += num_cleared
                reward_this_step += num_cleared * 0.1 # Continuous reward
                if cleared_info["combo"]:
                    reward_this_step += 1.0 # Combo reward
                
                self._unsettle_floating_fragments()
            else:
                # No more matches, chain reaction is over
                break

        self._update_particles()
        
        # --- 4. Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.score >= self.TARGET_SCORE:
            terminated = True
            reward_this_step += 100 # Win reward
        elif self._check_fill_condition():
            terminated = True
            reward_this_step -= 100 # Lose penalty
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time/step limit
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_difficulty(self):
        self.fall_speed = self.fall_speed_base + 0.01 * (self.steps // 200)
        self.spawn_rate = self.spawn_rate_base + 0.005 * (self.steps // 100)

    def _spawn_fragments(self):
        self.spawn_timer += self.spawn_rate
        if self.spawn_timer >= 1.0:
            self.spawn_timer -= 1.0
            
            x_pos = self.np_random.integers(0, self.SCREEN_WIDTH // self.FRAGMENT_WIDTH) * self.FRAGMENT_WIDTH
            y = -self.FRAGMENT_HEIGHT if self.gravity == 1 else self.SCREEN_HEIGHT
            
            # Ensure no immediate overlap at spawn
            is_overlapping = any(f.rect.x == x_pos and abs(f.rect.y - y) < self.FRAGMENT_HEIGHT for f in self.fragments if not f.settled)
            if is_overlapping:
                return

            color_idx = self.np_random.integers(0, len(self.FRAGMENT_COLORS))
            color = self.FRAGMENT_COLORS[color_idx]
            glyph = self.np_random.choice(list(self.GLYPHS))
            
            new_frag = Fragment(x_pos, y, self.FRAGMENT_WIDTH, self.FRAGMENT_HEIGHT, color, glyph)
            self.fragments.append(new_frag)

    def _move_and_settle_fragments(self):
        for frag in self.fragments:
            if frag.settled:
                continue

            # Apply gravity
            frag.rect.y += self.fall_speed * self.gravity
            
            # Check for settling on screen boundary
            if self.gravity == 1 and frag.rect.bottom >= self.SCREEN_HEIGHT:
                frag.rect.bottom = self.SCREEN_HEIGHT
                frag.settled = True
            elif self.gravity == -1 and frag.rect.top <= 0:
                frag.rect.top = 0
                frag.settled = True
            
            # Check for settling on other settled fragments
            if not frag.settled:
                for other in self.fragments:
                    if other == frag or not other.settled:
                        continue
                    
                    # Check for vertical collision
                    if frag.rect.colliderect(other.rect):
                        if self.gravity == 1:
                            frag.rect.bottom = other.rect.top
                        else: # gravity == -1
                            frag.rect.top = other.rect.bottom
                        frag.settled = True
                        break # Settled on one fragment, no need to check others
    
    def _find_and_clear_matches(self):
        if not self.fragments:
            return {"count": 0, "combo": False}

        to_remove = set()
        checked = set()
        
        settled_frags = [f for f in self.fragments if f.settled]

        for frag in settled_frags:
            if frag in checked:
                continue
            
            current_group = set()
            q = [frag]
            visited_in_search = {frag}

            while q:
                current_frag = q.pop(0)
                current_group.add(current_frag)
                checked.add(current_frag)

                # Check neighbors
                for neighbor in settled_frags:
                    if neighbor in visited_in_search or neighbor.color != current_frag.color:
                        continue
                    
                    # Check for adjacency (touching sides, not just corners)
                    is_adjacent = (
                        abs(current_frag.rect.centerx - neighbor.rect.centerx) < self.FRAGMENT_WIDTH + 2 and
                        abs(current_frag.rect.centery - neighbor.rect.centery) < self.FRAGMENT_HEIGHT + 2 and
                        (current_frag.rect.x == neighbor.rect.x or current_frag.rect.y == neighbor.rect.y)
                    )

                    if is_adjacent:
                         q.append(neighbor)
                         visited_in_search.add(neighbor)
            
            if len(current_group) >= 2:
                to_remove.update(current_group)

        if not to_remove:
            return {"count": 0, "combo": False}

        for frag in to_remove:
            self._create_explosion(frag.rect.centerx, frag.rect.centery, frag.color)
            if frag in self.fragments:
                self.fragments.remove(frag)
        
        return {"count": len(to_remove), "combo": len(to_remove) > 3}

    def _unsettle_floating_fragments(self):
        settled_frags = [f for f in self.fragments if f.settled]
        for frag in settled_frags:
            supported = False
            # Check if on floor/ceiling
            if (self.gravity == 1 and frag.rect.bottom >= self.SCREEN_HEIGHT) or \
               (self.gravity == -1 and frag.rect.top <= 0):
                supported = True
                continue

            # Check if supported by another fragment
            support_rect = frag.rect.move(0, self.gravity) # Check 1px below/above
            for other in settled_frags:
                if other != frag and support_rect.colliderect(other.rect):
                    supported = True
                    break
            
            if not supported:
                frag.settled = False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _create_explosion(self, x, y, color):
        num_particles = self.np_random.integers(15, 25)
        for _ in range(num_particles):
            self.particles.append(Particle(x, y, color))
    
    def _check_fill_condition(self):
        fill_line_y = 40
        for frag in self.fragments:
            if frag.settled:
                if self.gravity == 1 and frag.rect.top < fill_line_y:
                    return True
                if self.gravity == -1 and frag.rect.bottom > self.SCREEN_HEIGHT - fill_line_y:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw fragments
        for frag in self.fragments:
            # Draw glow effect
            glow_rect = frag.rect.inflate(6, 6)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*frag.color, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)

            # Draw main fragment
            pygame.draw.rect(self.screen, frag.color, frag.rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in frag.color), frag.rect, width=1, border_radius=3)

            # Draw glyph
            glyph_surface = self.font_glyph.render(frag.glyph, True, self.COLOR_BG)
            glyph_rect = glyph_surface.get_rect(center=frag.rect.center)
            self.screen.blit(glyph_surface, glyph_rect)

        # Draw particles
        for p in self.particles:
            alpha_color = (*p.color, max(0, min(255, int(p.life * 6))))
            s = pygame.Surface((p.radius*2, p.radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, int(p.radius), int(p.radius), alpha_color)
            self.screen.blit(s, (int(p.x - p.radius), int(p.y - p.radius)), special_flags=pygame.BLEND_RGBA_ADD)
            
    def _render_ui(self):
        # Draw score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw fill lines
        fill_line_y = 40
        s = pygame.Surface((self.SCREEN_WIDTH, 1), pygame.SRCALPHA)
        s.fill(self.COLOR_FILL_LINE)
        self.screen.blit(s, (0, fill_line_y))
        self.screen.blit(s, (0, self.SCREEN_HEIGHT - fill_line_y - 1))

        # Draw gravity indicator
        if self.gravity == 1:
            points = [(self.SCREEN_WIDTH - 30, 15), (self.SCREEN_WIDTH - 20, 25), (self.SCREEN_WIDTH - 10, 15)]
        else: # gravity == -1
            points = [(self.SCREEN_WIDTH - 30, 25), (self.SCREEN_WIDTH - 20, 15), (self.SCREEN_WIDTH - 10, 25)]
        pygame.draw.lines(self.screen, self.COLOR_UI_TEXT, False, points, 3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gravity": self.gravity,
            "fall_speed": self.fall_speed,
            "spawn_rate": self.spawn_rate
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The main loop is for demonstration and manual play, which requires a display.
    # The environment itself is headless as per the `os.environ` setting.
    # To run this example, you might need to comment out the `os.environ` line.
    
    # Re-enable display for manual play if not in a strictly headless environment
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Pygame setup for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Flip")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # No-op, Space released, Shift released

    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Press space
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    action = [0, 0, 0]
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0 # Release space
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # For manual play, we need to reset the action after one press
        if action[1] == 1:
            action[1] = 0

        # Render to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()