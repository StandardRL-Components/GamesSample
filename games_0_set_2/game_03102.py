import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fruit-catching arcade game.

    The player controls a basket at the bottom of the screen, moving it left and
    right to catch fruits that fall from the top. The game is divided into three
    timed stages of increasing difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move the basket. Catch the falling fruit!"
    )

    # User-facing game description
    game_description = (
        "Catch falling fruit in your basket to score points. Complete three "
        "timed stages to win, but be careful! If you miss 5 fruits in a "
        "single stage, or the timer runs out, it's game over."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_GRID = (125, 196, 225)
    COLOR_BASKET = (139, 69, 19)  # Saddle Brown
    COLOR_BASKET_RIM = (160, 82, 45)  # Sienna
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 128)
    COLOR_TIMER_WARN = (255, 100, 100)

    # Fruit Types & Colors
    FRUIT_COLORS = {
        'apple': (220, 20, 60),  # Crimson
        'banana': (255, 223, 0),  # Gold
        'grape': (106, 13, 173), # Violet
        'golden': (255, 215, 0) # Goldenrod
    }

    # Game Parameters
    BASKET_WIDTH = 100
    BASKET_HEIGHT = 20
    BASKET_Y_POS = SCREEN_HEIGHT - 40
    BASKET_SPEED = 8.0
    FRUIT_RADIUS = 12
    MAX_LIVES = 5
    STAGE_DURATION = 60 * FPS  # 60 seconds
    FRUITS_PER_STAGE = 10
    RISKY_CATCH_MARGIN = 0.25 # 25% from the edge


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)
        self.font_desc = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.basket_x = 0.0
        self.score = 0
        self.stage = 0
        self.total_fruits_caught = 0
        self.fruits_caught_this_stage = 0
        self.lives = 0
        self.timer = 0
        self.fruits = []
        self.particles = []
        self.game_over = False
        self.win = False
        self.message = ""
        self.message_timer = 0
        self.base_fruit_speed = 0.0
        self.fruit_spawn_timer = 0
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Initialize game state for a new episode
        self.basket_x = self.SCREEN_WIDTH / 2.0
        self.score = 0
        self.stage = 1
        self.total_fruits_caught = 0
        self.game_over = False
        self.win = False
        
        self._setup_stage()
        self._show_message(f"Stage {self.stage}", 2 * self.FPS)

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes variables for the current stage."""
        self.fruits_caught_this_stage = 0
        self.lives = self.MAX_LIVES
        self.timer = self.STAGE_DURATION
        self.fruits = []
        self.particles = []
        
        # Difficulty scaling
        self.base_fruit_speed = 2.0 + (self.stage - 1) * 0.75
        self.fruit_spawn_rate = max(20, 60 - (self.stage - 1) * 10)
        self.fruit_spawn_timer = self.fruit_spawn_rate

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.game_over:
            # After game is over, reset on next step
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        else: # No-op or other unused movements
            reward -= 0.01 # Small penalty for being idle

        self.basket_x = np.clip(self.basket_x, self.BASKET_WIDTH / 2, self.SCREEN_WIDTH - self.BASKET_WIDTH / 2)

        # --- Game Logic Update ---
        self.timer -= 1
        if self.message_timer > 0:
            self.message_timer -= 1

        # Fruit Spawning
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.rng.integers(self.fruit_spawn_rate, self.fruit_spawn_rate + 20)

        # Update Fruits
        for fruit in reversed(self.fruits):
            self._update_fruit_position(fruit)
            
            # Check for catch
            if self._check_catch(fruit):
                catch_reward = self._process_catch(fruit)
                reward += catch_reward
                self.fruits.remove(fruit)
            
            # Check for miss
            elif fruit['pos'][1] > self.SCREEN_HEIGHT + self.FRUIT_RADIUS:
                self.lives -= 1
                reward -= 50  # Penalty for losing a life
                self.fruits.remove(fruit)

        # Update Particles
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Check Game State ---
        # Stage progression
        if self.fruits_caught_this_stage >= self.FRUITS_PER_STAGE:
            if self.stage < 3:
                self.stage += 1
                reward += 50  # Stage completion bonus
                self._setup_stage()
                self._show_message(f"Stage {self.stage}", 2 * self.FPS)
            elif not self.win: # Just completed stage 3
                self.win = True
                self.game_over = True
                reward += 100 # Game win bonus
                self._show_message("You Win!", 5 * self.FPS)

        # Termination conditions
        if (self.timer <= 0 or self.lives <= 0) and not self.game_over:
            self.game_over = True
            reward -= 100 # Game over penalty
            self._show_message("Game Over", 5 * self.FPS)

        terminated = self.game_over
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_fruit_position(self, fruit):
        """Updates a fruit's position based on its type and game difficulty."""
        speed_multiplier = 1 + (self.total_fruits_caught // 10) * 0.1
        current_speed = self.base_fruit_speed * speed_multiplier
        
        fruit['pos'][1] += current_speed
        
        if fruit['type'] == 'golden': # Sinusoidal movement
            fruit['angle'] += fruit['freq']
            fruit['pos'][0] = fruit['start_x'] + math.sin(fruit['angle']) * fruit['amp']

    def _check_catch(self, fruit):
        """Checks if a fruit is caught by the basket."""
        basket_rect = pygame.Rect(
            self.basket_x - self.BASKET_WIDTH / 2,
            self.BASKET_Y_POS - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        return basket_rect.collidepoint(fruit['pos'][0], fruit['pos'][1])

    def _process_catch(self, fruit):
        """Handles the logic for a successful catch."""
        self.fruits_caught_this_stage += 1
        self.total_fruits_caught += 1
        
        base_reward = 2 if fruit['type'] == 'golden' else 1
        self.score += base_reward
        
        # Risky catch bonus
        dist_from_center = abs(fruit['pos'][0] - self.basket_x)
        if dist_from_center > (self.BASKET_WIDTH / 2) * (1 - self.RISKY_CATCH_MARGIN):
            bonus_reward = 5
            self.score += bonus_reward
            self._create_particles(fruit['pos'], fruit['color'], 20, is_bonus=True)
            return float(base_reward + bonus_reward)
        else:
            self._create_particles(fruit['pos'], fruit['color'], 10)
            return float(base_reward)

    def _spawn_fruit(self):
        """Creates a new fruit and adds it to the list."""
        x_pos = self.rng.uniform(self.FRUIT_RADIUS, self.SCREEN_WIDTH - self.FRUIT_RADIUS)
        
        # Determine fruit type
        fruit_type = 'apple' # Default
        if self.stage >= 2 and self.rng.random() < 0.25:
            fruit_type = 'golden'
        else:
            fruit_type = self.rng.choice(list(self.FRUIT_COLORS.keys() - {'golden'}))

        fruit = {
            'pos': [x_pos, -float(self.FRUIT_RADIUS)],
            'type': fruit_type,
            'color': self.FRUIT_COLORS[fruit_type],
            'radius': self.FRUIT_RADIUS if fruit_type != 'banana' else self.FRUIT_RADIUS*0.7
        }
        
        if fruit_type == 'golden':
            fruit['start_x'] = x_pos
            fruit['amp'] = self.rng.uniform(30, 80)
            fruit['freq'] = self.rng.uniform(0.05, 0.1)
            fruit['angle'] = self.rng.uniform(0, 2 * math.pi)

        self.fruits.append(fruit)

    def _create_particles(self, pos, color, count, is_bonus=False):
        """Spawns particles for visual effect."""
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.rng.integers(20, 40),
                'radius': self.rng.integers(2, 5),
                'color': color if not is_bonus else self.FRUIT_COLORS['golden']
            })

    def _show_message(self, text, duration):
        """Sets a message to be displayed on screen for a duration."""
        self.message = text
        self.message_timer = duration

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        self._render_messages()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "lives": self.lives,
            "timer": math.ceil(self.timer / self.FPS),
            "fruits_caught": self.total_fruits_caught,
        }

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_game_elements(self):
        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = int(fruit['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, fruit['color'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40.0))
            color_with_alpha = (*p['color'], alpha)
            try:
                # This function is picky about color format
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color_with_alpha)
            except (ValueError, TypeError):
                # Fallback if alpha blending fails
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))


        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_x - self.BASKET_WIDTH / 2,
            self.BASKET_Y_POS - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, basket_rect, 3, border_radius=5)

    def _render_ui(self):
        # Helper to render text with shadow
        def draw_text(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (10, 10))

        # Lives
        lives_text = "♥" * self.lives
        draw_text(lives_text, self.font_ui, (255, 80, 80), (self.SCREEN_WIDTH - 120, 10))

        # Timer
        time_left = math.ceil(self.timer / self.FPS)
        timer_color = self.COLOR_TIMER_WARN if time_left <= 10 and self.timer > 0 else self.COLOR_TEXT
        timer_text = f"Time: {time_left}"
        timer_surf = self.font_ui.render(timer_text, True, timer_color)
        draw_text(timer_text, self.font_ui, timer_color, (self.SCREEN_WIDTH // 2 - timer_surf.get_width() // 2, 10))

    def _render_messages(self):
        if self.message_timer > 0:
            alpha = 255
            if self.message_timer < self.FPS: # Fade out
                alpha = int(255 * (self.message_timer / self.FPS))

            shadow_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT_SHADOW)
            text_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT)
            
            shadow_surf.set_alpha(alpha)
            text_surf.set_alpha(alpha)

            pos = (
                self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2,
                self.SCREEN_HEIGHT // 2 - text_surf.get_height() // 2
            )
            self.screen.blit(shadow_surf, (pos[0] + 3, pos[1] + 3))
            self.screen.blit(text_surf, pos)
            
    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The environment is designed to be headless, but for human play,
    # you can try to create a display.
    render_for_human = True
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        render_for_human = False

    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Example ---
    if render_for_human:
        try:
            # Re-initialize with a display
            pygame.display.init()
            pygame.display.set_caption("Fruit Catcher")
            screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            
            obs, info = env.reset()
            terminated = False
            clock = pygame.time.Clock()
            
            print("\n" + "="*30)
            print("    FRUIT CATCHER DEMO")
            print("="*30)
            print(env.game_description)
            print("\n" + env.user_guide)
            print("="*30 + "\n")

            while not terminated:
                # Action mapping for human play
                action = [0, 0, 0] # [movement, space, shift]
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    action[0] = 3
                elif keys[pygame.K_RIGHT]:
                    action[0] = 4
                
                if keys[pygame.K_q]:
                    terminated = True

                obs, reward, terminated, truncated, info = env.step(action)
                
                # Render the observation to the display
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                clock.tick(env.FPS)
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")

        except pygame.error as e:
            print("\nPygame display could not be initialized. Running in headless mode.")
            print(f"Pygame error: {e}")
            render_for_human = False
        finally:
            env.close()

    if not render_for_human:
        print("Running a short headless episode test...")
        obs, info = env.reset(seed=42)
        done = False
        step_count = 0
        total_reward = 0
        while not done and step_count < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        print(f"Headless episode finished after {step_count} steps.")
        print(f"Final Info: {info}")
        print(f"Total Reward: {total_reward}")
        env.close()